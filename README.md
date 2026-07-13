![Project banner](docs/source/_static/mjlab-banner.jpg)

# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
  <a href="https://mujocolab.github.io/mjlab/"><img alt="docs" src="https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg" /></a>
  <a href="https://mujocolab.github.io/mjlab/nightly/"><img alt="benchmarks" src="https://img.shields.io/badge/nightly-blue" /></a>
</p>

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

## Kuavo S45 and RoboParty RPO

This working tree stays on the mjlab 1.0 API so the existing S45 tasks and the
RoboParty 3.x PPO configuration remain compatible. The RPO asset is registered
with these standard PPO tasks:

```bash
uv run list_envs
uv run train Mjlab-Velocity-Flat-RPO --env.scene.num-envs 4096
uv run train Mjlab-Tracking-Flat-RPO --env.scene.num-envs 768
```

For an S45 CSV motion, use the S45-specific converter (it trims legacy 28-DOF
exports to the 26 training joints):

```bash
uv run python src/mjlab/scripts/csv_to_npz_s45.py \
  --input-file /path/to/s45.csv --output-name s45_motion
```

The RPO tracking task defaults to
`src/mjlab/asset_zoo/robots/rpo/motions/yundong1.npz`; override
`--env.commands.motion.motion_file` to use another bundled NPZ. The original
RoboParty AMP/AttnEnc/Parkour runners are not registered here because they use
custom rsl-rl 3.3 modules that are separate from mjlab's standard PPO runner.

See [`docs/rpo_migration_notes.md`](docs/rpo_migration_notes.md) for the exact
asset conversion and MuJoCo height-ray procedure.

---

## Quick Start

mjlab requires an **NVIDIA GPU** for training (via MuJoCo Warp).
macOS is supported only for evaluation, which is significantly slower.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run the demo (no installation needed):

```bash
uvx --from mjlab --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@7c20a44bfed722e6415235792a1b247ea6b6a6d3" demo
```

This launches an interactive viewer with a pre-trained Unitree G1 agent tracking a reference dance motion in MuJoCo Warp.

> ❓ Having issues? See the [FAQ](https://mujocolab.github.io/mjlab/source/faq.html).

**Try in Google Colab (no local setup required):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb)

Launch the demo directly in your browser with an interactive Viser viewer.

---

## Installation

**From source:**

```bash
git clone https://github.com/mujocolab/mjlab.git
cd mjlab
uv run demo
```

**From PyPI:**

```bash
uv add mjlab "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@7c20a44bfed722e6415235792a1b247ea6b6a6d3"
```

A Dockerfile is also provided.

For full setup instructions, see the [Installation Guide](https://mujocolab.github.io/mjlab/source/installation.html).

---

## Training Examples

### 1. Kuavo S45 Velocity Tracking

Train the Kuavo S45 humanoid on rough terrain:

```bash
uv run train Mjlab-Velocity-Rough-KUAVO-S45 --env.scene.num-envs 1024
```

For multi-GPU training:

```bash
uv run train Mjlab-Velocity-Rough-KUAVO-S45 \
  --gpu-ids 0 1 2 3 4 5 6 \
  --env.scene.num-envs 1024
```

Flat-terrain evaluation:

```bash
uv run play Mjlab-Velocity-Flat-KUAVO-S45 --wandb-run-path your-org/mjlab/run-id
```

### 2. RPO Velocity Tracking

Train the RPO humanoid on rough terrain:

```bash
uv run train Mjlab-Velocity-Rough-RPO --env.scene.num-envs 1024
```

For multi-GPU training:

```bash
uv run train Mjlab-Velocity-Rough-RPO \
  --gpu-ids 0 1 2 3 4 5 6 \
  --env.scene.num-envs 1024
```

Flat-terrain evaluation:

```bash
uv run play Mjlab-Velocity-Flat-RPO --wandb-run-path your-org/mjlab/run-id
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/source/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

---

### 3. Motion Imitation

Train Kuavo S45 or RPO to mimic reference motions. mjlab uses
[WandB](https://wandb.ai) to manage reference motion datasets:

1. **Create a registry collection** in your WandB workspace named `Motions`

2. **Set your WandB entity**:
   ```bash
   export WANDB_ENTITY=your-organization-name
   ```

3. **Process and upload motion files**:
   ```bash
   MUJOCO_GL=egl uv run src/mjlab/scripts/csv_to_npz.py \
     --input-file /path/to/motion.csv \
     --output-name motion_name \
     --input-fps 30 \
     --output-fps 50 \
     --render  # Optional: generates preview video
   ```

> [!NOTE]
> For detailed motion preprocessing instructions, see the
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### Train and Play

```bash
uv run train Mjlab-Tracking-Flat-KUAVO-S45 --registry-name your-org/motions/motion-name --env.scene.num-envs 1024

uv run train Mjlab-Tracking-Flat-RPO --registry-name your-org/motions/motion-name --env.scene.num-envs 1024

uv run play Mjlab-Tracking-Flat-KUAVO-S45 --wandb-run-path your-org/mjlab/run-id

uv run play Mjlab-Tracking-Flat-RPO --wandb-run-path your-org/mjlab/run-id
```

---

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP **before** training.

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions.
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions.
```

> [!NOTE]
> When running motion-tracking tasks, add
> `--registry-name your-org/motions/motion-name` to the command.

---

## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

---

## Development

Run tests:

```bash
make test          # Run all tests
make test-fast     # Skip slow integration tests
```

Format code:

```bash
uvx pre-commit install
make format
```

Compile documentation locally:

```bash
uv pip install -r docs/requirements.txt
make docs
```

---

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

---

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
