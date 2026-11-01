import jax
import jax.numpy as jnp
import optax

from pendulumcontrol.simulation import PendulumSimulation
from pendulumcontrol.trajectory import QuadraticCost


def optimal_forces(
    simulation: PendulumSimulation,
    cost: QuadraticCost,
    states: jax.Array,
    target_state: jax.Array,
    max_force: float,
    discount: float = 0.99,
    horizon: int = 20,
    optimization_steps: int = 100,
    learning_rate: float = 0.1,
) -> jax.Array:
    """Optimizes a force sequence from each state by gradient descent through the
    simulation, and returns the first force of each sequence."""
    weights = discount ** jnp.arange(horizon)
    optimizer = optax.adam(learning_rate)

    def sequence_cost(parameters, initial_state):
        def advance(state, parameter):
            force = max_force * jnp.tanh(parameter)
            return simulation.step(state, force), cost(state, force, target_state)

        _, step_costs = jax.lax.scan(advance, initial_state, parameters)
        return jnp.sum(weights * step_costs)

    def solve(initial_state):
        def improve(carry, _):
            parameters, optimizer_state = carry
            gradients = jax.grad(sequence_cost)(parameters, initial_state)
            updates, optimizer_state = optimizer.update(gradients, optimizer_state)
            return (optax.apply_updates(parameters, updates), optimizer_state), None

        initial_parameters = jnp.zeros(horizon)
        (parameters, _), _ = jax.lax.scan(
            improve,
            (initial_parameters, optimizer.init(initial_parameters)),
            None,
            length=optimization_steps,
        )
        return max_force * jnp.tanh(parameters[0])

    return jax.jit(jax.vmap(solve))(states)
