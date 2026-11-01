from pathlib import Path

import jax
import jax.numpy as jnp

from pendulumcontrol import (
    PendulumSimulation,
    PolicyGradientTraining,
    StateControl,
    StateControlAgent,
    animate_simulation,
    plot_training,
    run_training,
    simulate,
)

REPORT_DIRECTORY = Path("reports/stabilize_pendulum")
INITIAL_STATE = jnp.array([0.5, 0.3, 0.0, 0.0])
SIMULATION_STEPS = 100

network_key, training_key = jax.random.split(jax.random.key(0))
simulation = PendulumSimulation()
agent = StateControlAgent(StateControl(network_key))

agent, histories = run_training(
    agent, simulation, [PolicyGradientTraining(updates=1500)], training_key
)

REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plot_training(histories, REPORT_DIRECTORY / "training.png")
trajectory = simulate(agent, simulation, INITIAL_STATE, SIMULATION_STEPS)
animate_simulation(trajectory, agent, simulation, REPORT_DIRECTORY / "simulation.gif")
