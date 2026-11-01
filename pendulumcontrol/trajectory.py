import equinox as eqx
import jax
import jax.numpy as jnp

from pendulumcontrol.agent import StateControlAgent
from pendulumcontrol.simulation import (
    ANGLE,
    ANGULAR_VELOCITY,
    POSITION,
    PendulumSimulation,
    wrap_angle,
)


class QuadraticCost(eqx.Module):
    position_weight: float = 0.1
    angle_weight: float = 1.0
    velocity_weight: float = 0.01
    angular_velocity_weight: float = 0.01
    force_weight: float = 0.000004

    def __call__(
        self, state: jax.Array, force: jax.Array, target_state: jax.Array
    ) -> jax.Array:
        error = state - target_state
        position_error, _, velocity_error, angular_velocity_error = error
        return (
            self.position_weight * position_error**2
            + self.angle_weight * wrap_angle(error[ANGLE]) ** 2
            + self.velocity_weight * velocity_error**2
            + self.angular_velocity_weight * angular_velocity_error**2
            + self.force_weight * force**2
        )


class Trajectory(eqx.Module):
    states: jax.Array
    sampled_controls: jax.Array
    applied_forces: jax.Array

    def step_costs(self, cost: QuadraticCost, target_state: jax.Array) -> jax.Array:
        return jax.vmap(cost, (0, 0, None))(
            self.states[:-1], self.applied_forces, target_state
        )


def simulate(
    agent: StateControlAgent,
    simulation: PendulumSimulation,
    initial_state: jax.Array,
    steps: int,
    key: jax.Array | None = None,
) -> Trajectory:
    """Runs the agent in closed loop, with exploration noise when a key is given."""

    def advance(state, step_key):
        if step_key is None:
            sampled = agent.control(state)
        else:
            sampled = agent.sample_control(state, step_key)
        force = agent.limit_force(sampled)
        return simulation.step(state, force), (state, sampled, force)

    step_keys = None if key is None else jax.random.split(key, steps)
    final_state, (states, sampled, forces) = jax.lax.scan(
        advance, initial_state, step_keys, length=steps
    )
    all_states = jnp.concatenate([states, final_state[None]])
    return Trajectory(all_states, sampled, forces)


def integrator_path(
    trajectory: Trajectory, simulation: PendulumSimulation
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns times, states and forces at every integrator step of a trajectory."""
    step_count = simulation.integrator_steps_per_control
    is_pulse = jnp.arange(step_count) < simulation.integrator_steps_with_force

    def advance(state, force):
        next_state = simulation.integrate(state, force, 1)
        next_state = next_state.at[ANGLE].set(wrap_angle(next_state[ANGLE]))
        return next_state, (state, force)

    def control_interval(state, force):
        return jax.lax.scan(advance, state, force * is_pulse)[1]

    states, forces = jax.vmap(control_interval)(
        trajectory.states[:-1], trajectory.applied_forces
    )
    states, forces = states.reshape(-1, states.shape[-1]), forces.reshape(-1)
    return simulation.integrator_time_step * jnp.arange(len(forces)), states, forces


def is_stabilized(
    trajectory: Trajectory,
    target_state: jax.Array,
    angle_tolerance: float = 0.1,
    angular_velocity_tolerance: float = 0.5,
    position_tolerance: float = 1.0,
    final_fraction: float = 0.1,
) -> jax.Array:
    """True if the state stays near the target for the end of the trajectory."""
    states = trajectory.states
    final_states = states[-max(1, int(len(states) * final_fraction)) :]
    errors = final_states - target_state
    return jnp.all(
        (jnp.abs(wrap_angle(errors[:, ANGLE])) < angle_tolerance)
        & (jnp.abs(errors[:, ANGULAR_VELOCITY]) < angular_velocity_tolerance)
        & (jnp.abs(errors[:, POSITION]) < position_tolerance)
    )
