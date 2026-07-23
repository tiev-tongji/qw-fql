import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['OGBENCH_DATA_DIR'] = '/mnt/data/user_workspace/zhouhongtu/ogbench_dataset'

import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
from datetime import datetime

# Use 'spawn' start method to avoid os.fork() warning with JAX.
# JAX is multithreaded and fork() copies its thread pool state, which can cause deadlocks.
# 'spawn' creates a fresh Python interpreter for each child process instead.
mp.set_start_method('spawn')

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_flag_dict, get_tb_video, setup_tensorboard

FLAGS = flags.FLAGS

# Global agent config overrides from command line (--agent.xxx=value).
_agent_overrides = {}

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'logs/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

# Multi-GPU multi-seed flags.
flags.DEFINE_list('gpu_ids', None, 'Comma-separated list of GPU IDs to use (e.g. 0,1,2,3,4,5,6,7).')
flags.DEFINE_list('seeds', None, 'Comma-separated list of random seeds (e.g. 0,1,2,3,4,5,6,7).')
flags.DEFINE_string('time_str', None, 'Experiment timestamp (auto-generated if not specified, shared across seeds in multi-seed mode).')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def _extract_agent_overrides(argv):
    """Extract --agent.xxx=value args from argv and return (clean_argv, overrides_dict)."""
    overrides = {}
    clean_argv = []
    for arg in argv:
        if arg.startswith('--agent.') and '=' in arg:
            key, value = arg[len('--agent.'):].split('=', 1)
            overrides[key] = value
        else:
            clean_argv.append(arg)
    return clean_argv, overrides


def _apply_agent_overrides(config, overrides):
    """Apply command-line overrides to the agent config dict."""
    import ast as ast_module
    for key, value in overrides.items():
        try:
            parsed = ast_module.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        config[key] = parsed
    return config


def _format_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def _run_single_seed(gpu_id, seed, cmd_args):
    """Run training for a single seed on a single GPU (subprocess target)."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    print(f'[GPU {gpu_id}] Starting seed={seed}: {" ".join(cmd_args)}', flush=True)
    proc = subprocess.run(cmd_args, env=env)
    print(f'[GPU {gpu_id}] seed={seed} finished with return code {proc.returncode}', flush=True)


def _train():
    """Core training logic, called after FLAGS are parsed and agent config is set."""
    config = FLAGS.agent
    agent_name = config['agent_name']
    alpha = config['alpha']
    tempe = config['qw_temperature']
    exp_name = f'{agent_name}_{alpha}_{tempe}'

    # Set up save directory: logs/env_name/agent_name/time/seed/
    time_str = FLAGS.time_str or datetime.now().strftime('%Y%m%d_%H%M%S')
    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, FLAGS.env_name, exp_name, time_str, f'seed{FLAGS.seed}'
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    writer = setup_tensorboard(log_dir=FLAGS.save_dir, config=get_flag_dict())
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=2)

    # Make environment and datasets.
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    total_steps = FLAGS.offline_steps + FLAGS.online_steps
    tag = f'[seed={FLAGS.seed}]'
    print(f'{tag} Training started | total_steps={total_steps} | save_dir={FLAGS.save_dir}', flush=True)

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in range(1, total_steps + 1):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])
            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)
            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            for k, v in train_metrics.items():
                writer.add_scalar(k, v, global_step=i)
            train_logger.log(train_metrics, step=i)

            # Print progress.
            progress = i / total_steps * 100
            elapsed = time.time() - first_time
            eta = elapsed / i * (total_steps - i) if i > 0 else 0
            print(
                f'{tag} step={i:>7d}/{total_steps} ({progress:5.1f}%) | '
                f'elapsed={_format_time(elapsed)} | eta={_format_time(eta)} | '
                f'epoch_time={train_metrics["time/epoch_time"]:.3f}s',
                flush=True,
            )

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_tb_video(renders=renders)
                eval_metrics['video'] = video

            for k, v in eval_metrics.items():
                if k == 'video':
                    writer.add_video('video', v, global_step=i, fps=15)
                else:
                    writer.add_scalar(k, v, global_step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()
    writer.close()
    print(f'{tag} Training finished | total_time={_format_time(time.time() - first_time)} | save_dir={FLAGS.save_dir}', flush=True)


def main(_):
    # Multi-seed mode: distribute seeds across GPUs using subprocesses.
    if FLAGS.seeds is not None and FLAGS.gpu_ids is not None:
        gpu_ids = [int(g.strip()) for g in FLAGS.gpu_ids]
        seeds = [int(s.strip()) for s in FLAGS.seeds]

        # Distribute seeds evenly across GPUs (round-robin).
        tasks = []
        for idx, seed in enumerate(seeds):
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            tasks.append((gpu_id, seed))

        print(f'[Launcher] Distributing {len(seeds)} seeds across {len(gpu_ids)} GPUs: {gpu_ids}', flush=True)
        for gpu_id, seed in tasks:
            print(f'  GPU {gpu_id} -> seed {seed}', flush=True)

        # Build base command for each subprocess (without gpu_ids/seeds to avoid recursion).
        base_cmd = [sys.executable, os.path.abspath(sys.argv[0])]
        # Generate shared time_str so all seeds go to the same directory.
        shared_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_cmd.append(f'--time_str={shared_time_str}')
        # Forward only explicitly set flags (not defaults) to avoid serialization issues.
        for flag_name in FLAGS:
            if flag_name in ('gpu_ids', 'seeds', 'agent', 'time_str'):
                continue
            if FLAGS[flag_name].present:
                base_cmd.append(f'--{flag_name}={getattr(FLAGS, flag_name)}')
        # Forward --agent with absolute path (FLAGS.agent is a ConfigDict, so extract from sys.argv).
        agent_path = None
        for arg in sys.argv:
            if arg.startswith('--agent='):
                agent_path = arg.split('=', 1)[1]
                break
        if agent_path is None:
            agent_path = 'agents/fql.py'  # Default config file.
        base_cmd.append(f'--agent={os.path.abspath(agent_path)}')
        # Forward agent config overrides (--agent.xxx=value).
        for key, value in _agent_overrides.items():
            base_cmd.append(f'--agent.{key}={value}')

        # Launch subprocesses.
        processes = []
        for gpu_id, seed in tasks:
            # Each subprocess gets a specific seed but no gpu_ids/seeds (single-seed mode).
            cmd = list(base_cmd) + [f'--seed={seed}']
            p = mp.Process(target=_run_single_seed, args=(gpu_id, seed, cmd))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        return

    # Single-seed mode.
    # Apply command-line overrides (--agent.xxx=value).
    if _agent_overrides:
        _apply_agent_overrides(FLAGS.agent, _agent_overrides)
        print(f'[seed={FLAGS.seed}] Agent config overrides: {_agent_overrides}', flush=True)

    _train()


if __name__ == '__main__':
    # Extract --agent.xxx overrides before absl parses flags.
    sys.argv, _agent_overrides = _extract_agent_overrides(sys.argv)
    app.run(main)
