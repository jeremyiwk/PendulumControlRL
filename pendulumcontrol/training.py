from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pendulumcontrol.agent import StateControl, StateControlAgent
from pendulumcontrol.optimal_control import optimal_forces
from pendulumcontrol.simulation import ANGLE, PendulumSimulation, wrap_angle
from pendulumcontrol.trajectory import (
    QuadraticCost,
    Trajectory,
    is_stabilized,
    simulate,
)


class StateRange(eqx.Module):
    """Uniform range of states around the target."""

    max_position: float = 0.5
    max_angle: float = 0.3
    max_velocity: float = 0.5
    max_angular_velocity: float = 0.5

    def sample(self, key: jax.Array, count: int, target: jax.Array) -> jax.Array:
        bounds = jnp.array(
            [
                self.max_position,
                self.max_angle,
                self.max_velocity,
                self.max_angular_velocity,
            ]
        )
        offsets = jax.random.uniform(key, (count, 4), minval=-bounds, maxval=bounds)
        states = target + offsets
        return states.at[:, ANGLE].set(wrap_angle(states[:, ANGLE]))


@dataclass
class TrainingHistory:
    name: str
    loss_label: str
    accuracy_label: str
    train_states: np.ndarray
    test_states: np.ndarray
    updates: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)
    live_trajectory: Trajectory | None = None
    gradient_norms: list[float] = field(default_factory=list)
    stretch_costs: list[float] = field(default_factory=list)
    perturbation_updates: list[int] = field(default_factory=list)

    def record(
        self,
        update: int,
        train_loss: jax.Array,
        test_loss: jax.Array,
        train_accuracy: jax.Array,
        test_accuracy: jax.Array,
    ) -> None:
        self.updates.append(update)
        self.train_loss.append(float(train_loss))
        self.test_loss.append(float(test_loss))
        self.train_accuracy.append(float(train_accuracy))
        self.test_accuracy.append(float(test_accuracy))


class TrainingMethod(Protocol):
    def train(
        self, agent: StateControlAgent, simulation: PendulumSimulation, key: jax.Array
    ) -> tuple[StateControlAgent, TrainingHistory]: ...


def run_training(
    agent: StateControlAgent,
    simulation: PendulumSimulation,
    methods: Sequence[TrainingMethod],
    key: jax.Array,
) -> tuple[StateControlAgent, list[TrainingHistory]]:
    histories = []
    for method, method_key in zip(
        methods, jax.random.split(key, len(methods)), strict=True
    ):
        agent, history = method.train(agent, simulation, method_key)
        histories.append(history)
        print(
            f"{history.name}: test loss {history.test_loss[-1]:.4f}, "
            f"test accuracy {history.test_accuracy[-1]:.3f}"
        )
    return agent, histories


def apply_gradients(models, gradients, optimizer, optimizer_state):
    parameters = eqx.filter(models, eqx.is_inexact_array)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, parameters)
    return eqx.apply_updates(models, updates), optimizer_state


@dataclass(frozen=True)
class OptimalControlPretraining:
    """Regression onto forces found by optimizing through the simulation."""

    cost: QuadraticCost = QuadraticCost()
    state_range: StateRange = StateRange()
    discount: float = 0.99
    train_state_count: int = 2000
    test_state_count: int = 250
    updates: int = 5000
    batch_size: int = 256
    learning_rate: float = 1e-3
    evaluation_interval: int = 100
    accuracy_tolerance: float = 0.1

    def train(
        self, agent: StateControlAgent, simulation: PendulumSimulation, key: jax.Array
    ) -> tuple[StateControlAgent, TrainingHistory]:
        train_key, test_key, key = jax.random.split(key, 3)
        target = agent.target_state

        def label(states):
            forces = optimal_forces(
                simulation, self.cost, states, target, agent.max_force, self.discount
            )
            return forces / agent.max_force

        train_states = self.state_range.sample(
            train_key, self.train_state_count, target
        )
        test_states = self.state_range.sample(test_key, self.test_state_count, target)
        train_labels = label(train_states)
        test_labels = label(test_states)

        def errors(network, states, labels):
            return jax.vmap(network)(jax.vmap(agent.network_input)(states)) - labels

        def loss(network, states, labels):
            return jnp.mean(errors(network, states, labels) ** 2)

        @eqx.filter_jit
        def evaluate(network, states, labels):
            within = jnp.abs(errors(network, states, labels)) < self.accuracy_tolerance
            return loss(network, states, labels), jnp.mean(within)

        optimizer = optax.adam(
            optax.cosine_decay_schedule(self.learning_rate, self.updates)
        )

        @eqx.filter_jit
        def update(network, optimizer_state, batch_key):
            indices = jax.random.randint(
                batch_key, (self.batch_size,), 0, self.train_state_count
            )
            gradients = eqx.filter_grad(loss)(
                network, train_states[indices], train_labels[indices]
            )
            return apply_gradients(network, gradients, optimizer, optimizer_state)

        history = TrainingHistory(
            "Optimal control pretraining",
            "Squared control error",
            f"Controls within {self.accuracy_tolerance:.0%} of optimal",
            np.asarray(train_states),
            np.asarray(test_states),
        )
        network = agent.network
        optimizer_state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
        for index, batch_key in enumerate(jax.random.split(key, self.updates + 1)):
            if index % self.evaluation_interval == 0 or index == self.updates:
                train_loss, train_accuracy = evaluate(
                    network, train_states, train_labels
                )
                test_loss, test_accuracy = evaluate(network, test_states, test_labels)
                history.record(
                    index, train_loss, test_loss, train_accuracy, test_accuracy
                )
            if index < self.updates:
                network, optimizer_state = update(network, optimizer_state, batch_key)
        return agent.with_network(network), history


@dataclass(frozen=True)
class TrajectoryTraining(ABC):
    """Shared training loop. Subclasses define one update from trajectories."""

    name: str = "Trajectory training"
    cost: QuadraticCost = QuadraticCost()
    state_range: StateRange = StateRange()
    discount: float = 0.99
    updates: int = 200
    trajectory_count: int = 128
    test_trajectory_count: int = 128
    steps: int = 20
    test_steps: int = 100
    learning_rate: float = 3e-4
    gradient_limit: float = 1.0
    evaluation_interval: int = 5
    angle_tolerance: float = 0.1
    angular_velocity_tolerance: float = 0.5

    @property
    def optimizer_steps_per_update(self) -> int:
        return 1

    def create_value_network(self, agent: StateControlAgent, key: jax.Array):
        return None

    @abstractmethod
    def update(
        self,
        models: tuple,
        optimizer_state,
        optimizer: optax.GradientTransformation,
        agent: StateControlAgent,
        simulation: PendulumSimulation,
        initial_states: jax.Array,
        update_index: jax.Array,
        key: jax.Array,
    ) -> tuple[tuple, object, Trajectory]: ...

    def collect(
        self,
        agent: StateControlAgent,
        simulation: PendulumSimulation,
        initial_states: jax.Array,
        key: jax.Array | None = None,
        steps: int | None = None,
    ) -> Trajectory:
        def run(state, run_key):
            return simulate(agent, simulation, state, steps or self.steps, run_key)

        if key is None:
            return jax.vmap(run, (0, None))(initial_states, None)
        return jax.vmap(run)(initial_states, jax.random.split(key, len(initial_states)))

    def create_optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.gradient_limit),
            optax.adam(
                optax.cosine_decay_schedule(
                    self.learning_rate, self.updates * self.optimizer_steps_per_update
                )
            ),
        )

    def evaluate(
        self,
        network: StateControl,
        agent: StateControlAgent,
        simulation: PendulumSimulation,
        test_states: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        batch = self.collect(
            agent.with_network(network), simulation, test_states, steps=self.test_steps
        )
        return self.measure(batch, agent.target_state)

    def measure(
        self, batch: Trajectory, target: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        step_costs = jax.vmap(Trajectory.step_costs, (0, None, None))
        stabilized = jax.vmap(is_stabilized, (0, None, None, None))
        return (
            jnp.mean(step_costs(batch, self.cost, target)),
            jnp.mean(
                stabilized(
                    batch,
                    target,
                    self.angle_tolerance,
                    self.angular_velocity_tolerance,
                )
            ),
        )

    def train(
        self, agent: StateControlAgent, simulation: PendulumSimulation, key: jax.Array
    ) -> tuple[StateControlAgent, TrainingHistory]:
        value_key, test_key, train_key, key = jax.random.split(key, 4)
        target = agent.target_state
        test_states = self.state_range.sample(
            test_key, self.test_trajectory_count, target
        )
        models = (agent.network, self.create_value_network(agent, value_key))
        optimizer = self.create_optimizer()
        optimizer_state = optimizer.init(eqx.filter(models, eqx.is_inexact_array))

        @eqx.filter_jit
        def update(models, optimizer_state, initial_states, index, update_key):
            models, optimizer_state, batch = self.update(
                models,
                optimizer_state,
                optimizer,
                agent,
                simulation,
                initial_states,
                index,
                update_key,
            )
            return models, optimizer_state, self.measure(batch, target)

        @eqx.filter_jit
        def evaluate(network):
            return self.evaluate(network, agent, simulation, test_states)

        history = TrainingHistory(
            self.name,
            "Mean cost per step",
            "Trajectories stabilized",
            np.asarray(
                self.state_range.sample(train_key, self.trajectory_count, target)
            ),
            np.asarray(test_states),
        )
        for index, update_key in enumerate(jax.random.split(key, self.updates)):
            state_key, step_key = jax.random.split(update_key)
            initial_states = self.state_range.sample(
                state_key, self.trajectory_count, target
            )
            models, optimizer_state, (train_loss, train_accuracy) = update(
                models, optimizer_state, initial_states, jnp.asarray(index), step_key
            )
            if (index + 1) % self.evaluation_interval == 0 or index == 0:
                test_loss, test_accuracy = evaluate(models[0])
                history.record(
                    index + 1, train_loss, test_loss, train_accuracy, test_accuracy
                )
        return agent.with_network(models[0]), history


def discounted_sum(values: jax.Array, discount: float) -> jax.Array:
    def accumulate(total, value):
        total = value + discount * total
        return total, total

    _, totals = jax.lax.scan(accumulate, jnp.zeros(()), values, reverse=True)
    return totals


@dataclass(frozen=True)
class PolicyGradientTraining(TrajectoryTraining):
    """Weights the log probability of sampled controls by the reward that follows."""

    name: str = "Policy gradient"
    learning_rate: float = 1e-3

    def update(
        self,
        models,
        optimizer_state,
        optimizer,
        agent,
        simulation,
        initial_states,
        update_index,
        key,
    ):
        network, value_network = models
        current = agent.with_network(network)
        batch = self.collect(current, simulation, initial_states, key)
        step_costs = jax.vmap(Trajectory.step_costs, (0, None, None))(
            batch, self.cost, agent.target_state
        )
        returns = jax.vmap(discounted_sum, (0, None))(-step_costs, self.discount)
        advantages = returns - returns.mean(axis=0, keepdims=True)
        advantages = advantages / (advantages.std() + 1e-8)

        def loss(network):
            log_probability = jax.vmap(
                jax.vmap(agent.with_network(network).log_probability)
            )(batch.states[:, :-1], batch.sampled_controls)
            return -jnp.mean(log_probability * advantages)

        gradients = eqx.filter_grad(loss)(network)
        models, optimizer_state = apply_gradients(
            models, (gradients, value_network), optimizer, optimizer_state
        )
        return models, optimizer_state, batch


@dataclass(frozen=True)
class ProximalPolicyTraining(TrajectoryTraining):
    """Several updates per batch, each limited to stay near the sampled policy."""

    name: str = "Proximal policy optimization"
    advantage_decay: float = 1.0
    update_limit: float = 0.2
    passes_per_batch: int = 1
    minibatch_count: int = 1
    learning_rate: float = 1e-3
    reward_scale: float = 1.0
    gradient_limit: float = 1e6
    value_loss_weight: float = 0.5
    value_hidden_size: int = 64
    value_hidden_layers: int = 2
    value_only_updates: int = 0
    normalize_advantages: bool = True

    @property
    def optimizer_steps_per_update(self) -> int:
        return self.passes_per_batch * self.minibatch_count

    def create_value_network(self, agent, key):
        return eqx.nn.MLP(
            len(agent.target_state),
            "scalar",
            self.value_hidden_size,
            self.value_hidden_layers,
            activation=jnp.tanh,
            key=key,
        )

    def update(
        self,
        models,
        optimizer_state,
        optimizer,
        agent,
        simulation,
        initial_states,
        update_index,
        key,
    ):
        return self.update_and_measure_gradient(
            models,
            optimizer_state,
            optimizer,
            agent,
            simulation,
            initial_states,
            update_index,
            key,
        )[:3]

    def update_and_measure_gradient(
        self,
        models,
        optimizer_state,
        optimizer,
        agent,
        simulation,
        initial_states,
        update_index,
        key,
    ):
        network, value_network = models
        policy_weight = jnp.where(update_index < self.value_only_updates, 0.0, 1.0)
        collect_key, shuffle_key = jax.random.split(key)
        current = agent.with_network(network)
        batch = self.collect(current, simulation, initial_states, collect_key)
        target = agent.target_state

        def state_values(state):
            return value_network(agent.network_input(state))

        def advantages_and_returns(trajectory):
            rewards = -self.reward_scale * trajectory.step_costs(self.cost, target)
            values = jax.vmap(state_values)(trajectory.states)
            errors = rewards + self.discount * values[1:] - values[:-1]
            advantages = discounted_sum(errors, self.discount * self.advantage_decay)
            return advantages, advantages + values[:-1]

        advantages, returns = jax.vmap(advantages_and_returns)(batch)
        states = batch.states[:, :-1].reshape(-1, agent.target_state.size)
        sampled = batch.sampled_controls.reshape(-1)
        old_log_probability = jax.vmap(current.log_probability)(states, sampled)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        def loss(models, indices):
            network, value_network = models
            batch_advantages = advantages[indices]
            if self.normalize_advantages:
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )
            log_probability = jax.vmap(agent.with_network(network).log_probability)(
                states[indices], sampled[indices]
            )
            ratio = jnp.exp(log_probability - old_log_probability[indices])
            limited = jnp.clip(ratio, 1 - self.update_limit, 1 + self.update_limit)
            policy_loss = -jnp.mean(jnp.minimum(ratio, limited) * batch_advantages)
            values = jax.vmap(value_network)(
                jax.vmap(agent.network_input)(states[indices])
            )
            value_loss = jnp.mean((values - returns[indices]) ** 2)
            return policy_weight * policy_loss + self.value_loss_weight * value_loss

        parameters, static = eqx.partition(models, eqx.is_inexact_array)

        def minibatch_update(carry, indices):
            parameters, optimizer_state = carry
            models = eqx.combine(parameters, static)
            gradients = eqx.filter_grad(loss)(models, indices)
            models, optimizer_state = apply_gradients(
                models, gradients, optimizer, optimizer_state
            )
            carry = eqx.filter(models, eqx.is_inexact_array), optimizer_state
            return carry, optax.tree.norm(gradients[0])

        minibatch_size = len(states) // self.minibatch_count
        orders = jax.vmap(
            lambda order_key: jax.random.permutation(order_key, len(states))
        )(jax.random.split(shuffle_key, self.passes_per_batch))
        indices = orders[:, : minibatch_size * self.minibatch_count].reshape(
            -1, minibatch_size
        )
        (parameters, optimizer_state), gradient_norms = jax.lax.scan(
            minibatch_update, (parameters, optimizer_state), indices
        )
        models = eqx.combine(parameters, static)
        return models, optimizer_state, batch, jnp.mean(gradient_norms)


@dataclass(frozen=True)
class LiveProximalPolicyTraining(ProximalPolicyTraining):
    """Learns during one continuous simulation that is never reset."""

    name: str = "Live proximal policy optimization"
    learning_rate: float = 3e-5
    normalize_advantages: bool = False
    perturbation_size: float = 0.0
    perturbation_cost: float = 1.0
    perturbation_gradient: float = 0.1
    perturbation_window: int = 50

    def create_optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.gradient_limit),
            optax.adam(self.learning_rate),
        )

    def perturb(self, network: StateControl, key: jax.Array) -> StateControl:
        parameters, static = eqx.partition(network, eqx.is_inexact_array)
        leaves, structure = jax.tree.flatten(parameters)
        keys = jax.random.split(key, len(leaves))
        leaves = [
            leaf + self.perturbation_size * jax.random.normal(leaf_key, leaf.shape)
            for leaf, leaf_key in zip(leaves, keys, strict=True)
        ]
        return eqx.combine(jax.tree.unflatten(structure, leaves), static)

    def train(
        self, agent: StateControlAgent, simulation: PendulumSimulation, key: jax.Array
    ) -> tuple[StateControlAgent, TrainingHistory]:
        value_key, test_key, start_key, key = jax.random.split(key, 4)
        target = agent.target_state
        test_states = self.state_range.sample(
            test_key, self.test_trajectory_count, target
        )
        state = self.state_range.sample(start_key, 1, target)
        models = (agent.network, self.create_value_network(agent, value_key))
        optimizer = self.create_optimizer()
        optimizer_state = optimizer.init(eqx.filter(models, eqx.is_inexact_array))

        @eqx.filter_jit
        def update(models, optimizer_state, state, index, update_key):
            models, optimizer_state, batch, gradient_norm = (
                self.update_and_measure_gradient(
                    models,
                    optimizer_state,
                    optimizer,
                    agent,
                    simulation,
                    state,
                    index,
                    update_key,
                )
            )
            return (
                models,
                optimizer_state,
                batch,
                gradient_norm,
                self.measure(batch, target),
            )

        @eqx.filter_jit
        def evaluate(network):
            return self.evaluate(network, agent, simulation, test_states)

        history = TrainingHistory(
            self.name,
            "Mean cost per step",
            "Stretches ending stabilized (train), runs stabilized (test)",
            np.asarray(state),
            np.asarray(test_states),
        )
        stretches = []
        recent_costs = deque(maxlen=self.perturbation_window)
        recent_gradients = deque(maxlen=self.perturbation_window)
        for index, update_key in enumerate(jax.random.split(key, self.updates)):
            step_key, perturbation_key = jax.random.split(update_key)
            models, optimizer_state, batch, gradient_norm, measures = update(
                models, optimizer_state, state, jnp.asarray(index), step_key
            )
            train_loss, train_accuracy = measures
            state = batch.states[:, -1]
            stretches.append(batch)
            recent_costs.append(float(train_loss))
            recent_gradients.append(float(gradient_norm))
            history.gradient_norms.append(float(gradient_norm))
            history.stretch_costs.append(float(train_loss))
            if (
                self.perturbation_size > 0
                and len(recent_costs) == self.perturbation_window
                and np.median(recent_costs) > self.perturbation_cost
                and np.median(recent_gradients) < self.perturbation_gradient
            ):
                network = self.perturb(models[0], perturbation_key)
                models = (network, models[1])
                history.perturbation_updates.append(index + 1)
                recent_costs.clear()
                recent_gradients.clear()
            if (index + 1) % self.evaluation_interval == 0 or index == 0:
                test_loss, test_accuracy = evaluate(models[0])
                history.record(
                    index + 1, train_loss, test_loss, train_accuracy, test_accuracy
                )
        history.train_states = np.concatenate(
            [np.asarray(stretch.states[0, :-1]) for stretch in stretches]
        )
        history.live_trajectory = Trajectory(
            jnp.concatenate(
                [stretch.states[0, :-1] for stretch in stretches] + [state]
            ),
            jnp.concatenate([stretch.sampled_controls[0] for stretch in stretches]),
            jnp.concatenate([stretch.applied_forces[0] for stretch in stretches]),
        )
        return agent.with_network(models[0]), history


@dataclass(frozen=True)
class SimulationGradientTraining(TrajectoryTraining):
    """Differentiates the discounted trajectory cost through the simulation itself."""

    name: str = "Simulation gradient"

    def update(
        self,
        models,
        optimizer_state,
        optimizer,
        agent,
        simulation,
        initial_states,
        update_index,
        key,
    ):
        network, value_network = models
        weights = self.discount ** jnp.arange(self.steps)

        def loss(network):
            current = agent.with_network(network)
            batch = self.collect(current, simulation, initial_states)
            step_costs = jax.vmap(Trajectory.step_costs, (0, None, None))(
                batch, self.cost, agent.target_state
            )
            return jnp.mean(step_costs * weights), batch

        gradients, batch = eqx.filter_grad(loss, has_aux=True)(network)
        models, optimizer_state = apply_gradients(
            models, (gradients, value_network), optimizer, optimizer_state
        )
        return models, optimizer_state, batch
