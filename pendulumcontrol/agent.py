from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from pendulumcontrol.simulation import ANGLE, wrap_angle


class StateControl(eqx.Module):
    """Maps a state to a control value in [-1, 1]."""

    layers: eqx.nn.MLP

    def __init__(
        self,
        key: jax.Array,
        state_size: int = 4,
        hidden_size: int = 64,
        hidden_layers: int = 2,
        activation: Callable = jnp.tanh,
    ):
        self.layers = eqx.nn.MLP(
            state_size,
            "scalar",
            hidden_size,
            hidden_layers,
            activation=activation,
            final_activation=jnp.tanh,
            key=key,
        )

    def __call__(self, state: jax.Array) -> jax.Array:
        return self.layers(state)


class StateControlAgent(eqx.Module):
    network: StateControl
    target_state: jax.Array = eqx.field(default_factory=lambda: jnp.zeros(4))
    state_scale: jax.Array = eqx.field(
        default_factory=lambda: jnp.array([2.0, 1.0, 3.0, 3.0])
    )
    max_force: float = 50.0
    exploration_noise: float = 0.2

    def network_input(self, state: jax.Array) -> jax.Array:
        error = state - self.target_state
        return error.at[ANGLE].set(wrap_angle(error[ANGLE])) / self.state_scale

    def control(self, state: jax.Array) -> jax.Array:
        return self.max_force * self.network(self.network_input(state))

    def sample_control(self, state: jax.Array, key: jax.Array) -> jax.Array:
        noise = self.max_force * self.exploration_noise * jax.random.normal(key)
        return self.control(state) + noise

    def log_probability(self, state: jax.Array, sampled: jax.Array) -> jax.Array:
        noise_scale = self.max_force * self.exploration_noise
        return -0.5 * ((sampled - self.control(state)) / noise_scale) ** 2

    def limit_force(self, force: jax.Array) -> jax.Array:
        return jnp.clip(force, -self.max_force, self.max_force)

    def with_network(self, network: StateControl) -> "StateControlAgent":
        return eqx.tree_at(lambda agent: agent.network, self, network)
