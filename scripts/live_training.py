from pathlib import Path

import jax

from pendulumcontrol import (
    LiveProximalPolicyTraining,
    PendulumSimulation,
    StateControl,
    StateControlAgent,
    Trajectory,
    animate_simulation,
    plot_live_run,
    plot_training,
    run_training,
)

REPORT_DIRECTORY = Path("reports/live_training")
SIMULATED_HOURS = 2
STEPS_PER_UPDATE = 20
ANIMATED_STEPS = 300

network_key, training_key = jax.random.split(jax.random.key(0))
simulation = PendulumSimulation()
agent = StateControlAgent(StateControl(network_key))
updates = round(
    SIMULATED_HOURS * 3600 / (STEPS_PER_UPDATE * simulation.control_interval)
)
method = LiveProximalPolicyTraining(
    updates=updates,
    steps=STEPS_PER_UPDATE,
    evaluation_interval=updates // 100,
    learning_rate=1e-3,
    normalize_advantages=True,
    perturbation_size=0.3,
    perturbation_gradient=0.5,
)

agent, histories = run_training(agent, simulation, [method], training_key)
history = histories[0]
live = history.live_trajectory
assert live is not None
print(f"Weight perturbations: {len(history.perturbation_updates)}")

REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plot_training(histories, REPORT_DIRECTORY / "training.png")
plot_live_run(history, simulation, REPORT_DIRECTORY / "live_run.png")
final_stretch = Trajectory(
    live.states[-ANIMATED_STEPS - 1 :],
    live.sampled_controls[-ANIMATED_STEPS:],
    live.applied_forces[-ANIMATED_STEPS:],
)
animate_simulation(final_stretch, agent, simulation, REPORT_DIRECTORY / "final_30s.gif")
