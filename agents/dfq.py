import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class DFQAgent(flax.struct.PyTreeNode):
    """Decoupled Flow Q-learning (DFQ) agent."""

    rng: Any
    network: Any
    alpha: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # VA flow loss.
        rng, bc_noise_rng = jax.random.split(rng)
        candi_num = self.config['candidate_num']
        bc_noises = jax.random.normal(bc_noise_rng, (batch_size, candi_num, action_dim))
        bc_obs = jnp.repeat(batch['observations'][:, None, ...], candi_num, axis=1)
        bc_actions = self.compute_bc_flow_actions(bc_obs, noises=bc_noises)

        candi_obs = jnp.repeat(batch['observations'][:, None, ...], candi_num+1, axis=1)
        candi_actions = jnp.concatenate([batch['actions'][:, None, ...], bc_actions], axis=1) # B, C+1, A
        candidate_qs = self.network.select('target_critic')(candi_obs, actions=candi_actions) # E, B, C+1
        candidate_qs = jax.lax.stop_gradient(candidate_qs.min(axis=0))  # Aggregate ensemble dimension -> B, C+1

        rng, sample_rng = jax.random.split(rng)
        candi_indices = jax.random.categorical(sample_rng, candidate_qs / self.config['va_temperature'])  # (B,)

        rng, va_x_rng, va_t_rng = jax.random.split(rng, 3)
        va_x_0 = jax.random.normal(va_x_rng, (batch_size, action_dim))
        va_x_1 = jnp.take_along_axis(candi_actions, candi_indices[:, None, None], axis=1).squeeze(1)
        va_x_1 = jax.lax.stop_gradient(va_x_1)
        va_t = jax.random.uniform(va_t_rng, (batch_size, 1))
        va_x_t = (1 - va_t) * va_x_0 + va_t * va_x_1
        va_vel = va_x_1 - va_x_0
        
        va_pred = self.network.select('actor_va_flow')(batch['observations'], va_x_t, va_t, params=grad_params)
        va_flow_loss = jnp.mean((va_pred - va_vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_target_va_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.  Normalize it to make the policy objective insensitive to the
        # reward / horizon-dependent scale of Q.  The straight-through clip
        # keeps the critic input in range while preserving actor gradients.
        actor_actions_raw = actor_actions

        def q_objective(actions):
            clipped_actions = actions + jax.lax.stop_gradient(
                jnp.clip(actions, -1, 1) - actions
            )
            qs = self.network.select('critic')(batch['observations'], actions=clipped_actions)
            q = jnp.min(qs, axis=0)
            q_loss_raw = -q.mean()

            if self.config.get('normalize_q_loss', True):
                q_scale = jax.lax.stop_gradient(
                    jnp.maximum(jnp.abs(q).mean(), self.config.get('q_scale_min', 1.0))
                )
            else:
                q_scale = jnp.asarray(1.0, dtype=q.dtype)

            q_loss = q_loss_raw / q_scale
            return q_loss, (q_loss_raw, q.mean(), q_scale)

        (q_loss, (q_loss_raw, q_mean, q_scale)), q_action_grad = jax.value_and_grad(
            q_objective, has_aux=True
        )(actor_actions_raw)

        # Adapt alpha using action-space gradient norms.  Both losses propagate
        # through the same one-step actor, so this is a stable and inexpensive
        # proxy for balancing their parameter-gradient contributions.
        distill_action_grad = 2.0 * (actor_actions_raw - target_flow_actions) / actor_actions_raw.size
        q_grad_norm = jnp.linalg.norm(q_action_grad)
        distill_grad_norm = jnp.linalg.norm(distill_action_grad)
        grad_eps = self.config.get('alpha_grad_eps', 1e-8)

        if self.config.get('adaptive_alpha', True):
            target_ratio = self.config.get('target_distill_q_grad_ratio', 1.0)
            alpha_target = target_ratio * q_grad_norm / (distill_grad_norm + grad_eps)
            alpha_target = jnp.clip(
                alpha_target,
                self.config.get('alpha_min', 0.01),
                self.config.get('alpha_max', 1000.0),
            )
            alpha_decay = self.config.get('alpha_ema_decay', 0.99)
            adaptive_alpha = alpha_decay * self.alpha + (1.0 - alpha_decay) * alpha_target
            adaptive_alpha = jnp.clip(
                adaptive_alpha,
                self.config.get('alpha_min', 0.01),
                self.config.get('alpha_max', 1000.0),
            )
            adaptive_alpha = jax.lax.stop_gradient(adaptive_alpha)
        else:
            alpha_target = jnp.asarray(self.config['alpha'], dtype=distill_loss.dtype)
            adaptive_alpha = jnp.asarray(self.config['alpha'], dtype=distill_loss.dtype)

        weighted_grad_ratio = adaptive_alpha * distill_grad_norm / (q_grad_norm + grad_eps)
        grad_cosine = jnp.sum(q_action_grad * distill_action_grad) / (
            q_grad_norm * distill_grad_norm + grad_eps
        )

        # Total loss.
        actor_loss = bc_flow_loss + va_flow_loss + adaptive_alpha * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)
        data_action_rate = jnp.mean((candi_indices == 0).astype(jnp.float32))

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'va_flow_loss': va_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q_loss_raw': q_loss_raw,
            'q_scale': q_scale,
            'q': q_mean,
            'alpha': adaptive_alpha,
            'alpha_target': alpha_target,
            'q_action_grad_norm': q_grad_norm,
            'distill_action_grad_norm': distill_grad_norm,
            'weighted_distill_q_grad_ratio': weighted_grad_ratio,
            'q_distill_grad_cosine': grad_cosine,
            'mse': mse,
            'data_action_rate': data_action_rate,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        self.target_update(new_network, 'actor_va_flow')

        return self.replace(network=new_network, alpha=info['actor/alpha'], rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_bc_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_target_va_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the VA flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('target_actor_va_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('target_actor_va_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_va_flow'] = encoder_module()
            encoders['target_actor_va_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_va_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_va_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_va_flow=(actor_va_flow_def, (ex_observations, ex_actions, ex_times)),
            target_actor_va_flow=(copy.deepcopy(actor_va_flow_def), (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        if encoders.get('actor_va_flow') is not None:
            # Add actor_va_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_va_flow_encoder'] = (encoders.get('actor_va_flow'), (ex_observations,))
        if encoders.get('target_actor_va_flow') is not None:
            # Add target_actor_va_flow_encoder to ModuleDict to make it separately callable.
            network_info['target_actor_va_flow_encoder'] = (encoders.get('target_actor_va_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_va_flow'] = params['modules_actor_va_flow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(
            rng,
            network=network,
            alpha=jnp.asarray(config['alpha'], dtype=jnp.float32),
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dfq',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # Fixed distillation coefficient when adaptive_alpha=False.
            adaptive_alpha=True,  # Balance Q and distillation gradients automatically.
            target_distill_q_grad_ratio=1.0,  # ||alpha * grad L_distill|| / ||grad L_Q||.
            alpha_min=0.01,  # Lower bound for the adaptive distillation coefficient.
            alpha_max=1000.0,  # Upper bound for the adaptive distillation coefficient.
            alpha_ema_decay=0.99,  # Smooth batch-to-batch adaptive-alpha changes.
            alpha_grad_eps=1e-8,
            candidate_num=4,
            va_temperature=0.5,
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            q_scale_min=1.0,  # Prevent excessive Q normalization near zero.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
