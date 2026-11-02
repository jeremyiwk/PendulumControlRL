from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes

from pendulumcontrol.agent import StateControlAgent
from pendulumcontrol.simulation import (
    ANGLE,
    ANGULAR_VELOCITY,
    POSITION,
    PendulumSimulation,
)
from pendulumcontrol.training import TrainingHistory
from pendulumcontrol.trajectory import Trajectory, integrator_path

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
LINE_WIDTH = 1.5
MAX_PLOTTED_STATES = 600

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": TEXT_SECONDARY,
    }
)


def add_legend(axes: Axes) -> None:
    axes.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1),
        ncols=4,
        borderaxespad=0,
        frameon=False,
    )


def plot_training(histories: Sequence[TrainingHistory], path: str | Path) -> None:
    figure, axes = plt.subplots(
        len(histories),
        3,
        figsize=(12, 3.4 * len(histories)),
        squeeze=False,
        layout="constrained",
    )
    for history, (loss_axes, accuracy_axes, state_axes) in zip(
        histories, axes, strict=True
    ):
        loss_axes.set_title(history.name)
        for series, label, color in (
            (history.train_loss, "Train", SERIES[0]),
            (history.test_loss, "Test", SERIES[1]),
        ):
            loss_axes.plot(
                history.updates, series, color=color, lw=LINE_WIDTH, label=label
            )
        loss_axes.set(xlabel="Update", ylabel=history.loss_label)
        add_legend(loss_axes)

        for series, label, color in (
            (history.train_accuracy, "Train", SERIES[0]),
            (history.test_accuracy, "Test", SERIES[1]),
        ):
            accuracy_axes.plot(
                history.updates, series, color=color, lw=LINE_WIDTH, label=label
            )
        accuracy_axes.set(
            xlabel="Update", ylabel=history.accuracy_label, ylim=(-0.02, 1.02)
        )
        add_legend(accuracy_axes)

        for states, label, color in (
            (history.train_states, "Train", SERIES[0]),
            (history.test_states, "Test", SERIES[1]),
        ):
            shown = states[:MAX_PLOTTED_STATES]
            state_axes.scatter(
                shown[:, ANGLE],
                shown[:, ANGULAR_VELOCITY],
                s=8,
                color=color,
                alpha=0.6,
                label=label,
            )
        state_axes.set(xlabel="Angle (rad)", ylabel="Angular velocity (rad/s)")
        state_axes.set_title("Sampled states")
        add_legend(state_axes)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_method_comparison(
    histories: Sequence[TrainingHistory], path: str | Path
) -> None:
    figure, (loss_axes, accuracy_axes) = plt.subplots(
        1, 2, figsize=(13, 3.8), layout="constrained"
    )
    for history, color in zip(histories, SERIES, strict=False):
        for axes, series in (
            (loss_axes, history.test_loss),
            (accuracy_axes, history.test_accuracy),
        ):
            axes.plot(
                history.updates, series, color=color, lw=LINE_WIDTH, label=history.name
            )
    loss_axes.set_title("Test cost")
    loss_axes.set(xlabel="Update", ylabel=histories[0].loss_label)
    loss_axes.set_yscale("log")
    accuracy_axes.set_title("Test stabilized")
    accuracy_axes.set(
        xlabel="Update", ylabel=histories[0].accuracy_label, ylim=(-0.02, 1.02)
    )
    handles, labels = loss_axes.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper right", ncols=len(histories))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_live_run(
    history: TrainingHistory,
    simulation: PendulumSimulation,
    path: str | Path,
    upright_angle: float = 0.2,
    minutes_per_bin: float = 5.0,
) -> None:
    assert history.live_trajectory is not None
    states = np.asarray(history.live_trajectory.states)
    steps_per_bin = round(minutes_per_bin * 60 / simulation.control_interval)
    bin_count = len(states) // steps_per_bin
    binned_angles = states[: bin_count * steps_per_bin, ANGLE].reshape(bin_count, -1)
    upright = np.mean(np.abs(binned_angles) < upright_angle, axis=1)
    bin_minutes = minutes_per_bin * (np.arange(bin_count) + 1)
    minutes = simulation.control_interval * np.arange(len(states)) / 60
    steps_per_update = (len(states) - 1) // len(history.stretch_costs)
    perturbation_minutes = (
        simulation.control_interval
        * steps_per_update
        * np.asarray(history.perturbation_updates)
        / 60
    )

    updates_per_bin = max(1, steps_per_bin // steps_per_update)
    costs = np.asarray(history.stretch_costs)
    cost_bin_count = len(costs) // updates_per_bin
    binned_costs = np.median(
        costs[: cost_bin_count * updates_per_bin].reshape(cost_bin_count, -1), axis=1
    )
    cost_minutes = minutes_per_bin * (np.arange(cost_bin_count) + 1)

    figure, (cost_axes, upright_axes, position_axes) = plt.subplots(
        1, 3, figsize=(15, 3.8), layout="constrained"
    )
    cost_axes.set_title("Live cost")
    cost_axes.plot(cost_minutes, binned_costs, color=SERIES[0], lw=LINE_WIDTH)
    cost_axes.set(
        xlabel="Simulated time (minutes)",
        ylabel=f"Median cost per step, each {minutes_per_bin:g} minutes",
        yscale="log",
    )
    upright_axes.set_title(f"Time within {upright_angle} rad of upright")
    upright_axes.plot(bin_minutes, upright, color=SERIES[0], lw=LINE_WIDTH)
    upright_axes.set(
        xlabel="Simulated time (minutes)",
        ylabel=f"Share of each {minutes_per_bin:g} minutes",
        ylim=(-0.02, 1.02),
    )
    position_axes.set_title("Fulcrum distance from target")
    position_axes.plot(
        minutes, np.abs(states[:, POSITION]) + 1e-3, color=SERIES[0], lw=LINE_WIDTH
    )
    position_axes.set(
        xlabel="Simulated time (minutes)", ylabel="Distance (m)", yscale="log"
    )
    for axes in (cost_axes, upright_axes, position_axes):
        for index, minute in enumerate(perturbation_minutes):
            axes.axvline(
                minute,
                color=TEXT_SECONDARY,
                lw=0.5,
                alpha=0.5,
                label="Weight perturbation" if index == 0 else None,
            )
    if len(perturbation_minutes) > 0:
        add_legend(position_axes)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def break_angle_jumps(angles: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Returns phase space points with gaps where the angle wraps around."""
    points = np.stack([angles, velocities], axis=1)
    jumps = np.flatnonzero(np.abs(np.diff(angles)) > np.pi) + 1
    return np.insert(points, jumps, np.nan, axis=0)


def animate_simulation(
    trajectory: Trajectory,
    agent: StateControlAgent,
    simulation: PendulumSimulation,
    path: str | Path,
    frames_per_second: int = 20,
) -> None:
    times, states, forces = (
        np.asarray(values) for values in integrator_path(trajectory, simulation)
    )
    positions, angles = states[:, POSITION], states[:, ANGLE]
    angular_velocities = states[:, ANGULAR_VELOCITY]
    target = np.asarray(agent.target_state)
    frame_interval = max(
        1, round(1 / (frames_per_second * simulation.integrator_time_step))
    )
    frames = np.arange(0, len(forces), frame_interval)

    figure, (pendulum_axes, phase_axes, force_axes) = plt.subplots(
        1, 3, figsize=(13, 3.8), layout="constrained"
    )
    reach = 1.2 * simulation.length
    pendulum_axes.set_title("Pendulum")
    pendulum_axes.set(
        xlim=(positions.min() - reach, positions.max() + reach),
        ylim=(-reach, reach),
        xlabel="Position (m)",
        yticks=[],
        aspect="equal",
    )
    pendulum_axes.axhline(0, color=GRID, lw=LINE_WIDTH)
    rod = pendulum_axes.plot(
        [], [], color=SERIES[0], lw=3, marker="o", markevery=[1], markersize=11
    )[0]
    fulcrum = pendulum_axes.plot([], [], "s", color=TEXT_PRIMARY, markersize=10)[0]
    clock = pendulum_axes.text(
        0.02, 0.04, "", transform=pendulum_axes.transAxes, color=TEXT_SECONDARY
    )

    full_path = break_angle_jumps(angles, angular_velocities)
    phase_axes.set_title("Phase space")
    phase_axes.plot(full_path[:, 0], full_path[:, 1], color=GRID, lw=LINE_WIDTH)
    phase_axes.scatter(
        target[ANGLE], target[ANGULAR_VELOCITY], s=40, color=TEXT_PRIMARY, marker="x"
    )
    phase_axes.set(xlabel="Angle (rad)", ylabel="Angular velocity (rad/s)")
    phase_path = phase_axes.plot([], [], color=SERIES[0], lw=LINE_WIDTH)[0]
    phase_point = phase_axes.plot([], [], "o", color=SERIES[0], markersize=7)[0]

    force_axes.set_title("Control")
    force_axes.axhline(agent.max_force, color=TEXT_SECONDARY, lw=0.8, ls="--")
    force_axes.axhline(-agent.max_force, color=TEXT_SECONDARY, lw=0.8, ls="--")
    force_axes.set(
        xlim=(times[0], times[-1]),
        ylim=(-1.15 * agent.max_force, 1.15 * agent.max_force),
        xlabel="Time (s)",
        ylabel="Force on fulcrum (N)",
    )
    force_path = force_axes.plot([], [], color=SERIES[0], lw=LINE_WIDTH)[0]

    def draw_frame(frame: int):
        position, angle = positions[frame], angles[frame]
        rod.set_data(
            [position, position + simulation.length * np.sin(angle)],
            [0, simulation.length * np.cos(angle)],
        )
        fulcrum.set_data([position], [0])
        clock.set_text(f"{times[frame]:.1f} s")
        path_so_far = break_angle_jumps(
            angles[: frame + 1], angular_velocities[: frame + 1]
        )
        phase_path.set_data(path_so_far[:, 0], path_so_far[:, 1])
        phase_point.set_data([angle], [angular_velocities[frame]])
        force_path.set_data(times[: frame + 1], forces[: frame + 1])
        return rod, fulcrum, clock, phase_path, phase_point, force_path

    animation = FuncAnimation(figure, draw_frame, frames=frames, blit=True)
    animation.save(path, writer=PillowWriter(fps=frames_per_second), dpi=100)
    plt.close(figure)
