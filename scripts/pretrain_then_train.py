from pathlib import Path

import jax

from pendulumcontrol import (
    OptimalControlPretraining,
    PendulumSimulation,
    ProximalPolicyTraining,
    StateControl,
    StateControlAgent,
    plot_training,
    run_training,
)

REPORT_DIRECTORY = Path("reports/pretrain_then_train")

network_key, training_key = jax.random.split(jax.random.key(0))
simulation = PendulumSimulation()
agent = StateControlAgent(StateControl(network_key))

agent, histories = run_training(
    agent,
    simulation,
    [
        OptimalControlPretraining(),
        ProximalPolicyTraining(updates=200, learning_rate=1e-4, value_only_updates=50),
    ],
    training_key,
)

REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plot_training(histories, REPORT_DIRECTORY / "training.png")
