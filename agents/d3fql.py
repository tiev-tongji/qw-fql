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


class D3FQLAgent(flax.struct.PyTreeNode):
    """D^3FQL agent: Flow Q-Learning with Dual-flow Decoupled Distillation."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def ensemble_stats(qs, kappa):
        """Compute ensemble mean, std, and LCB (mean - kappa * std) over the ensemble axis."""
        q_mean = qs.mean(axis=0)
        q_std = qs.std(axis=0)
        lcb = q_mean - kappa * q_std
        return q_mean, q_std, lcb

    def critic_loss(self, batch, grad_params, rng):
        """Compute the D^3FQL critic loss (LCB-aggregated bootstrap target)."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        _, _, next_q = self.ensemble_stats(next_qs, self.config['critic_lcb_kappa'])

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
        """Compute the D^3FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss (anchor flow).
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Candidate set: {dataset action} U {BC-flow samples} (support-locked proposal).
        rng, noise_rng = jax.random.split(rng)
        num_cand = self.config['bc_candidates']
        noises = jax.random.normal(noise_rng, (batch_size, num_cand, action_dim))
        n_observations = jnp.repeat(jnp.expand_dims(batch['observations'], 1), num_cand, axis=1)
        bc_actions = self.compute_bc_flow_actions(n_observations, noises=noises)

        cand_actions = jnp.concatenate([jnp.expand_dims(batch['actions'], 1), bc_actions], axis=1)  # (B, N+1, A)
        cand_obs = jnp.concatenate([jnp.expand_dims(batch['observations'], 1), n_observations], axis=1)

        # Pessimistic value oracle: LCB of the target critic ensemble.
        cand_qs = self.network.select('target_critic')(cand_obs, actions=cand_actions)  # (E, B, N+1)
        q_mean, q_std, lcb = self.ensemble_stats(cand_qs, self.config['lcb_kappa'])  # (B, N+1) each
        lcb = jax.lax.stop_gradient(lcb)

        # Soft Boltzmann weights over candidates.
        lam = jax.lax.stop_gradient(1.0 / (jnp.abs(lcb).mean() + 1e-6))
        weights = jax.nn.softmax(lam * lcb / self.config['qw_temperature'], axis=1)  # (B, N+1)
        weights = jax.lax.stop_gradient(weights)

        # QW flow loss: weighted conditional flow matching over ALL candidates.
        num_targets = num_cand + 1
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, num_targets, action_dim))
        x_1 = cand_actions  # (batch_size, N+1, action_dim)
        t = jax.random.uniform(t_rng, (batch_size, num_targets, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        flat_obs = jnp.repeat(batch['observations'], num_targets, axis=0)
        flat_x_t = x_t.reshape(batch_size * num_targets, action_dim)
        flat_t = t.reshape(batch_size * num_targets, 1)
        flat_vel = vel.reshape(batch_size * num_targets, action_dim)
        pred = self.network.select('actor_qw_flow')(flat_obs, flat_x_t, flat_t, params=grad_params)
        per_candidate_loss = jnp.mean((pred - flat_vel) ** 2, axis=-1).reshape(batch_size, num_targets)
        qw_flow_loss = jnp.mean(jnp.sum(weights * per_candidate_loss, axis=1))

        # Distillation losses (shared noise for both flows).
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)

        target_bc_flow_actions = self.compute_bc_flow_actions(batch['observations'], noises=noises)
        target_qw_flow_actions = self.compute_qw_flow_actions(batch['observations'], noises=noises)

        distill_bc_per = jnp.mean((actor_actions - target_bc_flow_actions) ** 2, axis=-1)  # (B,)
        distill_qw_per = jnp.mean((actor_actions - target_qw_flow_actions) ** 2, axis=-1)  # (B,)

        # State-dependent mixing coefficient eta(s) from ensemble disagreement.
        state_std = jax.lax.stop_gradient(q_std.mean(axis=1))  # (B,)
        conf = jax.lax.stop_gradient(jnp.exp(-state_std / (state_std.mean() + 1e-6)))  # (B,) in (0, 1]
        if self.config['adaptive_distill']:
            eta = self.config['beta'] * conf  # (B,)
        else:
            eta = jnp.full((batch_size,), self.config['beta'])
        distill_loss = jnp.mean((1.0 - eta) * distill_bc_per + eta * distill_qw_per)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('target_critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam_q = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam_q * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + qw_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)
        ess = jnp.mean(1.0 / jnp.sum(weights ** 2, axis=1))  # Effective sample size of the soft weights.

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'qw_flow_loss': qw_flow_loss,
            'distill_bc_flow_loss': distill_bc_per.mean(),
            'distill_qw_flow_loss': distill_qw_per.mean(),
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
            'qw_ess': ess,
            'conf_mean': conf.mean(),
            'eta_mean': eta.mean(),
            'q_std_mean': q_std.mean(),
            'lcb_mean': lcb.mean(),
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

        return self.replace(network=new_network, rng=new_rng), info

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
    def compute_qw_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the QW (improvement) flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_qw_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_qw_flow')(observations, actions, t, is_encoded=True)
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
            encoders['actor_qw_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_ensembles'],
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_qw_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_qw_flow'),
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
            actor_qw_flow=(actor_qw_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        if encoders.get('actor_qw_flow') is not None:
            # Add actor_qw_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_qw_flow_encoder'] = (encoders.get('actor_qw_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='d3fql',  # Agent name.
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
            num_ensembles=5,  # Number of Q-ensemble members (2 = clipped double-Q).
            critic_lcb_kappa=1.0,  # Pessimism for the bootstrap target (kappa=1.0 ~= ensemble min; 0.0 = mean).
            alpha=10.0,  # Overall distillation coefficient (need to be tuned for each environment).
            beta=0.9,  # Max mixing coefficient for the QW flow (eta(s) <= beta).
            bc_candidates=10,  # Number of BC action candidates.
            qw_temperature=1.0,  # Boltzmann temperature for the soft candidate weights (smaller = sharper).
            lcb_kappa=1.0,  # Pessimism for candidate weighting (0 = plain mean; ~=1.2 mimics the min with 5 members).
            adaptive_distill=True,  # Whether to modulate eta(s) per state by ensemble disagreement.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
