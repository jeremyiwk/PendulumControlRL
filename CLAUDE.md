# PendulumControlRL

Learned neural network control of an inverted pendulum on a moving fulcrum. Experiment code, so tests can stay light.

## Commands

- Install: `uv sync`, then `git config core.hooksPath .githooks`
- Test: `uv run pytest`
- Check, also run by the pre-commit hook: `uv run ruff format --check . && uv run ruff check . && uv run ty check`
- Run an experiment: `uv run python scripts/<name>.py`

## Layout

- `pendulumcontrol/simulation.py`: `PendulumSimulation`, Diffrax physics
- `pendulumcontrol/agent.py`: `StateControl` network, `StateControlAgent`
- `pendulumcontrol/trajectory.py`: `simulate`, `Trajectory`, `QuadraticCost`, `is_stabilized`
- `pendulumcontrol/optimal_control.py`: `optimal_forces`, labels for pretraining only
- `pendulumcontrol/training.py`: `TrainingMethod` protocol and its implementations
- `pendulumcontrol/plotting.py`: training plots and simulation GIF
- `scripts/`: experiments that write to `reports/<script name>/`

## Conventions

- Very few comments and docstrings. Names should explain the code.
- No jargon or abbreviations in names: `advantages`, not `adv`; `ProximalPolicyTraining`, not `PPO`.
- Keep code concise. Prefer plain control flow to new abstractions.
- State is `[position, angle, velocity, angular velocity]`. Angle 0 is upright and angles stay in [-pi, pi).
- The control is a force pulse on the fulcrum along x, lasting `pulse_duration` in every `control_interval` seconds. `simulation.step` advances one whole control interval, so trajectory steps are control intervals.
- The network maps state to control, and it alone acts at runtime. No hand-designed controllers such as LQR.
- Only `agent.network` is trained. The other agent fields are fixed settings.
