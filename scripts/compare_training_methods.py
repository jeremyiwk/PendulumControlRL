from pathlib import Path

import jax

from pendulumcontrol import (
    PendulumSimulation,
    PolicyGradientTraining,
    ProximalPolicyTraining,
    SimulationGradientTraining,
    StateControl,
    StateControlAgent,
    plot_method_comparison,
    run_training,
)

REPORT_DIRECTORY = Path("reports/compare_training_methods")
UPDATES = 1500

network_key, training_key = jax.random.split(jax.random.key(0))
simulation = PendulumSimulation()
agent = StateControlAgent(StateControl(network_key))

histories = []
for method in (
    PolicyGradientTraining(updates=UPDATES),
    ProximalPolicyTraining(updates=UPDATES),
    SimulationGradientTraining(updates=UPDATES),
):
    _, method_histories = run_training(agent, simulation, [method], training_key)
    histories.extend(method_histories)

REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
plot_method_comparison(histories, REPORT_DIRECTORY / "comparison.png")
