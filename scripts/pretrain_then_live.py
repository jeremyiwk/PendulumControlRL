import dataclasses
from pathlib import Path

import jax

from pendulumcontrol import (
    LiveProximalPolicyTraining,
    PendulumSimulation,
    PolicyGradientTraining,
    StateControl,
    StateControlAgent,
    Trajectory,
    animate_simulation,
    plot_live_run,
    plot_training,
    run_training,
)

REPORT_DIRECTORY = Path("reports/pretrain_then_live")
SIMULATED_HOURS = 8
STEPS_PER_UPDATE = 20
LIVE_EXPLORATION_NOISE = 0.1
ANIMATED_STEPS = 150

network_key, pretraining_key, live_key = jax.random.split(jax.random.key(0), 3)
simulation = PendulumSimulation()
agent = StateControlAgent(StateControl(network_key))
agent, pretraining_histories = run_training(
    agent, simulation, [PolicyGradientTraining(updates=1500)], pretraining_key
)

updates = round(
    SIMULATED_HOURS * 3600 / (STEPS_PER_UPDATE * simulation.control_interval)
)
live_agent = dataclasses.replace(agent, exploration_noise=LIVE_EXPLORATION_NOISE)
method = LiveProximalPolicyTraining(
    updates=updates, steps=STEPS_PER_UPDATE, evaluation_interval=updates // 100
)
live_agent, live_histories = run_training(live_agent, simulation, [method], live_key)
history = live_histories[0]
live = history.live_trajectory
assert live is not None

REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plot_training(pretraining_histories + live_histories, REPORT_DIRECTORY / "training.png")
plot_live_run(
    history, simulation, REPORT_DIRECTORY / "live_run.png", minutes_per_bin=15.0
)
for name, start in (("live_start.gif", 0), ("live_end.gif", -ANIMATED_STEPS - 1)):
    states = live.states[start:][: ANIMATED_STEPS + 1]
    stretch = Trajectory(
        states,
        live.sampled_controls[start:][:ANIMATED_STEPS],
        live.applied_forces[start:][:ANIMATED_STEPS],
    )
    animate_simulation(stretch, live_agent, simulation, REPORT_DIRECTORY / name)
