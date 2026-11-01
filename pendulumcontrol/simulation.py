import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

POSITION, ANGLE, VELOCITY, ANGULAR_VELOCITY = 0, 1, 2, 3


def wrap_angle(angle: jax.Array) -> jax.Array:
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


class PendulumSimulation(eqx.Module):
    """Pendulum on a fulcrum that moves along x, driven by a force on the fulcrum.

    State is (position, angle, velocity, angular velocity). Angle 0 is upright.
    """

    fulcrum_mass: float = 1.0
    pendulum_mass: float = 0.1
    length: float = 1.0
    gravity: float = 9.81
    fulcrum_friction: float = 0.1
    pendulum_damping: float = 0.01
    integrator_time_step: float = 0.005
    control_interval: float = 0.1
    pulse_duration: float = 0.01
    solver: diffrax.AbstractSolver = diffrax.Tsit5()

    @property
    def integrator_steps_per_control(self) -> int:
        return round(self.control_interval / self.integrator_time_step)

    @property
    def integrator_steps_with_force(self) -> int:
        return round(self.pulse_duration / self.integrator_time_step)

    def derivative(
        self, time: jax.typing.ArrayLike, state: jax.Array, force: jax.Array
    ):
        _, angle, velocity, angular_velocity = state
        sine, cosine = jnp.sin(angle), jnp.cos(angle)
        damping_term = self.pendulum_damping * angular_velocity
        acceleration = (
            force
            - self.fulcrum_friction * velocity
            + self.pendulum_mass
            * sine
            * (self.length * angular_velocity**2 - self.gravity * cosine)
            + cosine * damping_term / self.length
        ) / (self.fulcrum_mass + self.pendulum_mass * sine**2)
        angular_acceleration = (
            self.gravity * sine - cosine * acceleration
        ) / self.length - damping_term / (self.pendulum_mass * self.length**2)
        return jnp.stack(
            [velocity, angular_velocity, acceleration, angular_acceleration]
        )

    def integrate(self, state: jax.Array, force: jax.Array, steps: int) -> jax.Array:
        if steps == 0:
            return state
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.derivative),
            self.solver,
            t0=0.0,
            t1=steps * self.integrator_time_step,
            dt0=self.integrator_time_step,
            y0=state,
            args=force,
            max_steps=2 * steps,
        )
        assert solution.ys is not None
        return solution.ys[-1]

    def step(self, state: jax.Array, force: jax.Array) -> jax.Array:
        """Advances one control interval: a force pulse, then coasting with no force."""
        pulse_steps = self.integrator_steps_with_force
        coasting_steps = self.integrator_steps_per_control - pulse_steps
        state = self.integrate(state, force, pulse_steps)
        state = self.integrate(state, jnp.zeros_like(force), coasting_steps)
        return state.at[ANGLE].set(wrap_angle(state[ANGLE]))
