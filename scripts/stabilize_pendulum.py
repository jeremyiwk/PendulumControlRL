import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp

from pendulumcontrol import (
    PendulumSimulation,
    PolicyGradientTraining,
    QuadraticCost,
    StateControl,
    StateControlAgent,
    StateRange,
    animate_simulation,
    integrator_path,
    plot_training,
    run_training,
    simulate,
)

REPORT_DIRECTORY = Path("reports/stabilize_pendulum")
INITIAL_STATE = jnp.array([0.5, 0.3, 0.0, 0.0])
CONTROL_INTERVAL = 0.55
PULSE_DURATION = 0.005
MAX_FORCE = 500.0
TAP_FORCE = 50.0
EXPLORATION_FORCE = 20.0
TEST_STEPS = round(10 / CONTROL_INTERVAL)
SIMULATION_STEPS = round(20 / CONTROL_INTERVAL)

network_key, training_key = jax.random.split(jax.random.key(0))
simulation = PendulumSimulation(
    control_interval=CONTROL_INTERVAL, pulse_duration=PULSE_DURATION
)
agent = StateControlAgent(
    StateControl(network_key),
    max_force=MAX_FORCE,
    exploration_noise=EXPLORATION_FORCE / MAX_FORCE,
)
full_range_training = PolicyGradientTraining(
    name="Full range",
    updates=1000,
    steps=4,
    test_steps=TEST_STEPS,
    learning_rate=1e-4,
    cost=QuadraticCost(force_weight=0.01 / MAX_FORCE**2),
    angle_tolerance=0.4,
    angular_velocity_tolerance=2.0,
)
small_tilt_training = dataclasses.replace(
    full_range_training,
    name="Small tilts",
    state_range=StateRange(
        max_position=0.1, max_angle=0.1, max_velocity=0.2, max_angular_velocity=0.2
    ),
    learning_rate=3e-4,
)

agent, histories = run_training(
    agent, simulation, [small_tilt_training, full_range_training], training_key
)
tapping_agent = dataclasses.replace(agent, tap_force=TAP_FORCE)

REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plot_training(histories, REPORT_DIRECTORY / "training.png")
trajectory = simulate(tapping_agent, simulation, INITIAL_STATE, SIMULATION_STEPS)
_, states, _ = integrator_path(trajectory, simulation)
late_peak = float(jnp.abs(states[len(states) // 2 :, 1]).max())
print(f"Peak angle over the last 10 s with {TAP_FORCE:.0f} N taps: {late_peak:.3f} rad")
animate_simulation(
    trajectory, tapping_agent, simulation, REPORT_DIRECTORY / "simulation.gif"
)
