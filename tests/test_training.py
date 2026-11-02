import jax
import jax.numpy as jnp
import pytest

from pendulumcontrol import (
    LiveProximalPolicyTraining,
    OptimalControlPretraining,
    PendulumSimulation,
    PolicyGradientTraining,
    ProximalPolicyTraining,
    SimulationGradientTraining,
    StateControl,
    StateControlAgent,
    animate_simulation,
    integrator_path,
    is_stabilized,
    plot_live_run,
    plot_training,
    run_training,
    simulate,
)

KEY = jax.random.key(0)
SIMULATION = PendulumSimulation()
AGENT = StateControlAgent(StateControl(KEY, hidden_size=16))
INITIAL_STATE = jnp.array([0.0, 0.2, 0.0, 0.0])


def test_agent_force_stays_within_limit():
    trajectory = simulate(AGENT, SIMULATION, INITIAL_STATE, 20, KEY)
    assert trajectory.states.shape == (21, 4)
    assert jnp.all(jnp.abs(trajectory.applied_forces) <= AGENT.max_force)


@pytest.mark.parametrize(
    "method",
    [
        OptimalControlPretraining(
            updates=2, train_state_count=8, test_state_count=4, batch_size=4
        ),
        PolicyGradientTraining(updates=2, trajectory_count=4, steps=10, test_steps=10),
        ProximalPolicyTraining(updates=2, trajectory_count=4, steps=10, test_steps=10),
        SimulationGradientTraining(
            updates=2, trajectory_count=4, steps=10, test_steps=10
        ),
    ],
)
def test_training_method_changes_network(method, tmp_path):
    trained, histories = run_training(AGENT, SIMULATION, [method], KEY)
    network_input = AGENT.network_input(INITIAL_STATE)
    assert trained.network(network_input) != AGENT.network(network_input)
    plot_training(histories, tmp_path / "training.png")
    assert (tmp_path / "training.png").exists()


def test_tap_force_rounds_applied_force():
    agent = StateControlAgent(StateControl(KEY), max_force=500.0, tap_force=50.0)
    forces = jnp.array([20.0, 30.0, -80.0, 900.0])
    assert jnp.allclose(
        jax.vmap(agent.limit_force)(forces), jnp.array([0.0, 50.0, -100.0, 500.0])
    )


def test_policy_gradient_learns_to_stabilize():
    trained, _ = PolicyGradientTraining(updates=1500).train(
        StateControlAgent(StateControl(KEY)), SIMULATION, KEY
    )
    trajectory = simulate(trained, SIMULATION, INITIAL_STATE, 100)
    assert is_stabilized(trajectory, trained.target_state)


def test_integrator_path_applies_force_only_during_pulses():
    trajectory = simulate(AGENT, SIMULATION, INITIAL_STATE, 3)
    times, states, forces = integrator_path(trajectory, SIMULATION)
    steps = SIMULATION.integrator_steps_per_control
    assert len(times) == len(states) == len(forces) == 3 * steps
    assert jnp.all(forces[SIMULATION.integrator_steps_with_force : steps] == 0)
    assert jnp.allclose(states[steps], trajectory.states[1])


def test_live_training_is_one_continuous_run(tmp_path):
    method = LiveProximalPolicyTraining(updates=4, steps=5, test_steps=5)
    _, history = method.train(AGENT, SIMULATION, KEY)
    live = history.live_trajectory
    assert live is not None
    assert live.states.shape == (21, 4)
    next_states = jax.vmap(SIMULATION.step)(live.states[:-1], live.applied_forces)
    assert jnp.allclose(next_states, live.states[1:], atol=1e-4)
    assert not history.perturbation_updates
    plot_live_run(history, SIMULATION, tmp_path / "live.png", minutes_per_bin=0.01)


def test_live_training_perturbs_weights_when_triggered():
    method = LiveProximalPolicyTraining(
        updates=4,
        steps=5,
        test_steps=5,
        perturbation_size=0.1,
        perturbation_cost=-1.0,
        perturbation_gradient=float("inf"),
        perturbation_window=2,
    )
    _, history = method.train(AGENT, SIMULATION, KEY)
    assert history.perturbation_updates == [2, 4]


def test_animation_is_written(tmp_path):
    trajectory = simulate(AGENT, SIMULATION, INITIAL_STATE, 10)
    animate_simulation(trajectory, AGENT, SIMULATION, tmp_path / "simulation.gif")
    assert (tmp_path / "simulation.gif").exists()
