import jax.numpy as jnp
import pytest

from pendulumcontrol import PendulumSimulation, wrap_angle


def test_upright_state_is_an_equilibrium():
    state = PendulumSimulation().step(jnp.zeros(4), jnp.zeros(()))
    assert jnp.allclose(state, jnp.zeros(4))


def test_pendulum_falls_away_from_upright():
    state = PendulumSimulation().step(jnp.array([0.0, 0.1, 0.0, 0.0]), jnp.zeros(()))
    assert state[1] > 0.1
    assert state[3] > 0.0


def test_force_accelerates_fulcrum_and_tips_pendulum_backward():
    state = PendulumSimulation().step(jnp.zeros(4), jnp.array(5.0))
    assert state[2] > 0.0
    assert state[3] < 0.0


def test_frictionless_energy_and_momentum_are_conserved():
    simulation = PendulumSimulation(fulcrum_friction=0.0, pendulum_damping=0.0)
    masses = simulation.fulcrum_mass, simulation.pendulum_mass
    length, gravity = simulation.length, simulation.gravity

    def conserved(state):
        _, angle, velocity, angular_velocity = state
        horizontal = velocity + length * angular_velocity * jnp.cos(angle)
        vertical = length * angular_velocity * jnp.sin(angle)
        energy = (
            0.5 * masses[0] * velocity**2
            + 0.5 * masses[1] * (horizontal**2 + vertical**2)
            + masses[1] * gravity * length * jnp.cos(angle)
        )
        momentum = masses[0] * velocity + masses[1] * horizontal
        return float(energy), float(momentum)

    start = state = jnp.array([0.0, 1.0, 0.5, 0.0])
    for _ in range(200):
        state = simulation.step(state, jnp.zeros(()))
    assert conserved(state) == pytest.approx(conserved(start), abs=1e-3)


def test_pulse_delivers_force_for_its_duration_only():
    simulation = PendulumSimulation(fulcrum_friction=0.0, pendulum_damping=0.0)

    def momentum(state):
        horizontal = state[2] + simulation.length * state[3] * jnp.cos(state[1])
        return (
            simulation.fulcrum_mass * state[2] + simulation.pendulum_mass * horizontal
        )

    state = jnp.array([0.0, 0.2, 0.1, 0.3])
    change = momentum(simulation.step(state, jnp.array(50.0))) - momentum(state)
    assert float(change) == pytest.approx(50.0 * simulation.pulse_duration, abs=1e-5)


def test_integrator_step_is_much_smaller_than_control_interval():
    simulation = PendulumSimulation()
    assert simulation.integrator_steps_per_control == 20
    assert simulation.integrator_steps_with_force == 2


def test_wrap_angle_stays_in_range():
    assert float(wrap_angle(jnp.array(3 * jnp.pi / 2))) == pytest.approx(-jnp.pi / 2)
