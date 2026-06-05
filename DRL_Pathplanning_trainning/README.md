# DRL Cartesian Path Planning Training

**Layer 1: Frame-Only DRL Training** — Pure Python, optional PyBullet visualization.

This project trains Deep Reinforcement Learning agents (DDPG, SAC, TD3) to generate
optimal Cartesian waypoint paths in 3-D space. The agent learns to navigate a virtual
point from a start position to a target position using only normalized 3-D delta actions.
The output is a waypoint list that can later be consumed by a separate ROS2 / MoveIt
execution layer.

---

## Project Overview

This project implements **Layer 1** of a two-layer Cartesian path planning system:

- **Layer 1** (this project): Frame-only DRL training. A point-agent learns to generate
  waypoint paths in a 3-D Cartesian workspace. No robot model, no physics simulation,
  no IK/FK, no MoveIt.
- **Layer 2** (separate ROS2 package): Robot execution. MoveIt receives the waypoint
  list, handles IK, collision checking, joint limits, trajectory generation, and robot
  execution.

### What This Project Is NOT

This project does **not** train robot control. It does not use:
- Robot URDF files or robot model loading
- RoLE FK/IK or any kinematic library
- Robot joints, joint limits, or IK solvers
- PyBullet robot simulation (physics, dynamics, joint control, robot collision)
- MoveIt during training
- ROS/ROS2 during training

**PyBullet visualization** is optional and used only to visualise the abstract
Cartesian scene (workspace box, frames, obstacle, path). It does **not** run any
physics simulation, IK/FK, joint control, or robot dynamics.

---

## System Architecture

```
DRL_Pathplanning_trainning (Layer 1)
│
├── Gymnasium CartesianPathPlanningEnv
│   ├── Virtual point in 3-D Cartesian workspace
│   ├── Normalized action [ax, ay, az] in [-1, 1]^3
│   ├── delta = action * action_step (m)
│   └── next_pos = current_pos + delta
│
├── Stable-Baselines3
│   ├── DDPG (actor-critic, action noise)
│   ├── SAC  (actor-critic, entropy-based)
│   └── TD3  (twin Q-networks, delayed policy)
│
├── model.zip  (trained policy)
│
├── PyBullet Viewer (optional visualization)
│   ├── Workspace box wireframe
│   ├── Target sampling region
│   ├── base_link frame (large, world origin at [0,0,0])
│   ├── Start sphere (green, 0.01 m radius) + small XYZ axes
│   ├── Target sphere (yellow, 0.01 m radius) + small XYZ axes
│   ├── Agent point sphere + frame
│   ├── Small obstacle (red transparent, Collision-Free mode)
│   │   └── Label: OBSTACLE\ncenter: [...]\nsize: [...]
│   ├── Black block (solid black, fixed object)
│   ├── Plane (gray transparent, below workspace)
│   └── Path line strip
│
└── waypoint list  (x, y, z, distance, action, reward, flags)
        │
        ▼  (Layer 2 — separate ROS2 / MoveIt package)
├── MoveIt
│   ├── IK solving (joint angles from Cartesian waypoints)
│   ├── Collision checking against robot model
│   ├── Joint limit validation
│   ├── Trajectory planning (time-parameterized)
│   └── Robot execution
```

---

## Repository Structure

```
DRL_Pathplanning_trainning/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── config/
│   ├── environment.yaml         # Single source of truth for all environment parameters
│   └── experiments/
│       ├── ddpg_default.yaml
│       ├── sac_default.yaml
│       └── td3_default.yaml
│
├── src/drl_pathplanning/
│   ├── __init__.py
│   ├── config.py                  # YAML config loader + typed dataclasses
│   ├── spaces.py                  # Action/observation space helpers
│   ├── geometry.py                 # Box collision, segment-box, sampling
│   ├── reward.py                  # Reward function
│   │
│   ├── envs/
│   │   ├── __init__.py           # Gymnasium registration
│   │   └── cartesian_path_env.py # CartesianPathPlanningEnv
│   │
│   ├── visualization/              # Optional PyBullet visualization
│   │   ├── __init__.py
│   │   ├── colors.py             # RGBA colour constants
│   │   ├── debug_shapes.py       # PyBullet shape helpers (boxes, frames, lines)
│   │   └── pybullet_viewer.py   # PyBulletPathPlanningViewer class
│   │
│   ├── wrappers/
│   │   └── normalize.py          # (reserved — VecNormalize is disabled)
│   │
│   ├── callbacks/
│   │   └── training_callbacks.py # EpisodeCallback, BestModelCallback
│   │
│   └── utils/
│       ├── seed.py               # seed_everything()
│       ├── paths.py              # Project root, run dirs, model resolution
│       ├── logger.py             # Config/summary logging helpers
│       └── trajectory.py         # Waypoint/Trajectory classes + CSV/JSON I/O
│
├── Training/
│   ├── train_ddpg.py              # DDPG training
│   ├── train_sac.py              # SAC training
│   ├── train_td3.py              # TD3 training
│   └── train_all.py              # Sequential training runner
│
├── Evaluation/
│   ├── predict_static.py          # Fixed start/target prediction
│   ├── predict_random.py          # Random target evaluation
│   ├── evaluate_model.py          # Quantitative model evaluation
│   ├── export_waypoints.py       # Export waypoints to CSV/JSON
│   ├── plot_training.py          # Plot training curves
│   ├── test_pybullet_env.py       # PyBullet environment test + manual mode
│   ├── check_real_setup_pybullet.py   # PyBullet environment geometry verification
│   ├── run_start_to_random_target_pybullet.py  # Start-to-target visual test
│   └── visualize_rollout.py        # Visualize trained model with PyBullet
│
├── tests/
│   ├── test_config_loader.py       # Config loading, validation, and half-extent tests
│   ├── test_env_check.py          # SB3 env_checker validation
│   ├── test_observation.py        # Observation space tests
│   ├── test_reward.py             # Reward function unit tests
│   ├── test_collision.py          # Geometry / collision tests
│   ├── test_visualization_import.py # PyBullet viewer import / headless tests
│   └── test_environment_collision_logic.py  # Termination logic and collision tests
│
├── scripts/
│   ├── create_venv.sh            # Linux/macOS venv setup (.venv)
│   ├── create_venv.ps1           # Windows PowerShell venv setup (.venv)
│   ├── create_env_drl.sh         # Linux/macOS venv setup (env-drl)
│   ├── create_env_drl.ps1        # Windows PowerShell venv setup (env-drl)
│   └── check_install.py          # Package availability check
│
└── Data/
    └── Training/                  # Training outputs (gitignored)
```

---

## Installation

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python    | 3.10+   | Tested on 3.10 |
| CUDA      | 11.8/12.x | Optional — GPU training |

### Virtual Environment

The recommended virtual environment location is **outside** the project directory:

```bash
~/env-drl/bin/python -m venv ~/env-drl
source ~/env-drl/bin/activate
pip install -r requirements.txt
```

Activate before running any training or evaluation script:

```bash
source ~/env-drl/bin/activate
```

### Quick Setup (Linux / macOS — project-local venv)

```bash
cd DRL_Pathplanning_trainning
bash scripts/create_venv.sh
source .venv/bin/activate
```

### Quick Setup (Windows PowerShell)

```powershell
cd DRL_Pathplanning_trainning
powershell -ExecutionPolicy Bypass -File scripts/create_venv.ps1
.venv\Scripts\Activate.ps1
```

### Manual Setup

```bash
# 1. Navigate to the project directory
cd DRL_Pathplanning_trainning

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:
#.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.1 support (for GPU training)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Verify the installation
python scripts/check_install.py
```

---

## Optional PyBullet Visualization

PyBullet can be enabled to visualise the abstract Cartesian path planning environment
in a 3-D GUI window. It is used **only** as a viewer -- no URDF, no robot model, no
IK/FK, no joint control, no physics simulation. Training logic does not depend on
PyBullet; the environment runs headless without it.

### What is Visualised

| Element | Description |
|---------|-------------|
| Workspace box | Transparent wireframe + solid AABB bounding the search space |
| Target sampling region | Teal wireframe showing where targets are sampled |
| base_link frame | Large XYZ axis frame at world origin [0, 0, 0] (axis length 0.10 m) |
| Start sphere | Green solid sphere at the episode start position (radius 0.01 m) |
| Start frame | Small XYZ axis frame at the episode start position |
| Target sphere | Yellow solid sphere at the current target position (radius 0.01 m) |
| Target frame | Small XYZ axis frame at the current target position |
| Agent point | Green sphere + small XYZ frame showing current position |
| Obstacle | Red transparent box (Collision-Free mode only) with info label showing center and size |
| Obstacle info label | Multi-line text: "OBSTACLE\ncenter: [x, y, z]\nsize: [dx, dy, dz]" |
| Path | Yellow line strip showing the agent's trajectory |

### Enabling Visualization

```python
# Option 1: enable_visualization parameter
env = CartesianPathPlanningEnv(
    env_mode="Default",
    enable_visualization=True,
    visualization_config={"gui": True},
)

# Option 2: render_mode='human'
env = CartesianPathPlanningEnv(env_mode="Default", render_mode="human")
```

### Running Visualization Scripts

```bash
# Test the environment with random actions
python Evaluation/test_pybullet_env.py --mode Default --steps 200 --action-mode random --gui true

# Test with greedy (optimal) actions
python Evaluation/test_pybullet_env.py --mode Collision-Free --steps 200 --action-mode greedy --gui true

# Visualize a trained model rollout
python Evaluation/visualize_rollout.py \
    --model Data/Training/.../final_model.zip \
    --algorithm DDPG \
    --mode Default \
    --start 0.0 -0.3 0.2 \
    --target 0.4 -0.4 0.5 \
    --gui true
```

### Configuration

All environment parameters are defined in `config/environment.yaml` — the single source of truth for training, evaluation, and visualization scripts. Algorithm hyperparameters remain in `config/experiments/*.yaml`.

To override the default config, pass `--config path/to/config.yaml` to any script.

```yaml
# config/environment.yaml
project:
  name: DRL_Pathplanning_trainning
  robot_name: FRAME_ONLY
  unit: meter

start:
  mode: fixed
  position: [0.35, -0.33, 0.06]   # metres

workspace:
  name: search_region
  min: [-0.2, -0.7, 0.0]
  max: [0.6, 0.2, 0.6]

target_region:
  enabled: true
  random_target: true
  min: [0.0, -0.6, 0.05]
  max: [0.5, -0.2, 0.5]
  fixed_target: [0.3, -0.4, 0.25]

obstacle:
  enabled: true
  name: table
  type: box
  center: [0.0, -0.365, 0.18]   # metres (mid-height = z_bottom + height/2)
  size: [0.6, 0.4, 0.36]       # full sizes; half_extent = size / 2 in code

environment:
  mode: Collision-Free        # "Default" or "Collision-Free"
  action_step: 0.01          # metres per normalised action step
  max_steps: 200              # max steps per episode
  target_threshold: 0.01     # success threshold (metres)

reward:
  progress_scale: 10.0
  distance_scale: 1.0
  step_penalty: 0.01
  path_length_scale: 0.1
  success_bonus: 100.0
  collision_penalty: 100.0
  workspace_penalty: 100.0

visualization:
  enabled: true
  gui: true
  hide_debug_ui: true        # hide default PyBullet panels
  show_workspace: true
  show_target_region: true
  show_start_frame: true
  show_target_frame: true
  show_agent_frame: true
  show_obstacle: true
  show_path: true
  show_ground_plane: true
  show_labels: true

  camera:
    distance: 1.2
    yaw: -60
    pitch: -35
    target: [0.2, -0.25, 0.25]

  ground:
    z: 0.0
    center: [0.2, -0.25, 0.0]
    size: [1.0, 1.0]

  style:
    frame_axis_length: 0.05    # metres
    agent_radius: 0.015       # metres
    path_line_width: 4

training:
  algorithm: DDPG
  total_timesteps: 500000
  seed: 42
  device: auto
  vec_normalize: false

evaluation:
  seed: 42
  num_episodes: 100
  export_waypoints: true
```

**Important**: `obstacle.size` defines the full box dimensions. The `half_extent` is computed
automatically in code as `size / 2`. Do not store `half_extent` in the YAML.

### Checking the Frame Environment in PyBullet

The training environment uses **pure geometry-based collision detection** (AABB slab test in `geometry.py`).  PyBullet is **never used for collision, reward, or done-signal computation** — it is only a visualisation tool.

Before training, verify that the environment geometry matches your physical workspace by running the environment check script:

```bash
# Launch with default environment config
python Evaluation/check_real_setup_pybullet.py

# Manual terminal commands
python Evaluation/check_real_setup_pybullet.py --manual true --steps 200

# Automated greedy movement
python Evaluation/check_real_setup_pybullet.py --manual false --steps 50
```

All environment parameters are defined in `config/environment.yaml` — the single source of truth for training, evaluation, and visualization. To adapt it to your installation, edit `config/environment.yaml` or pass overrides as CLI arguments:

```yaml
workspace:
  min: [-0.2, -0.7, 0.0]   # metres
  max: [0.6, 0.2, 0.6]

target_region:
  min: [0.0, -0.6, 0.05]
  max: [0.5, -0.2, 0.5]

start:
  position: [0.35, -0.33, 0.06]

obstacle:                    # table / collision object
  enabled: true
  center: [0.0, -0.365, 0.18]
  size: [0.6, 0.4, 0.36]   # full sizes (converted to half-extent in code)

In **manual mode**, use these terminal commands to move the agent frame-by-frame and inspect coordinates:

| Command | Action |
|---------|--------|
| `x+` / `x-` | Move +x / -x |
| `y+` / `y-` | Move +y / -y |
| `z+` / `z-` | Move +z / -z |
| `r` | Reset episode |
| `q` | Quit |

Use manual mode to verify:
- Coordinate directions match your robot's tool frame
- Workspace boundary encloses the reachable volume
- Target region covers the desired goal area
- Obstacle/table position and z-height are correct
- All values match your physical setup

Full-featured environment test with all parameters passed as CLI arguments:

```bash
python Evaluation/test_pybullet_env.py \
  --mode Collision-Free \
  --workspace-min -0.2 -0.7 0.0 \
  --workspace-max 0.6 0.2 0.6 \
  --start 0.35 -0.33 0.06 \
  --target-region-min 0.0 -0.6 0.05 \
  --target-region-max 0.5 -0.2 0.5 \
  --obstacle-center 0.0 -0.365 0.18 \
  --obstacle-size 0.6 0.4 0.36 \
  --action-mode manual \
  --gui true
```

---

## Start-to-Random-Target PyBullet Test

This is a **visual sanity check**, not training. It moves the frame-only agent from the configured start point to either a fixed or random target inside `target_region` using fixed `0.01 m` greedy steps, draws the yellow path line, and logs start, target, step count, and path length.

**Before training**, use this to verify that the PyBullet scene geometry matches your physical setup:
- Coordinate directions match your robot's tool frame
- Workspace boundary encloses the reachable volume
- Target region covers the desired goal area
- Obstacle/table position and z-height are correct

### Setup

```bash
# Create and activate the isolated virtual environment
bash scripts/create_env_drl.sh
source ~/env-drl/bin/activate

# Verify all packages are installed
python scripts/check_install.py
```

### Running the test

**Default run — 100 episodes, GUI, clear path each episode:**

```bash
python Evaluation/run_start_to_random_target_pybullet.py
```

is equivalent to:

```bash
python Evaluation/run_start_to_random_target_pybullet.py \
    --config config/environment.yaml \
    --gui true \
    --episodes 100 \
    --step-size 0.01 \
    --sleep 0.08 \
    --pause-between-episodes 0.5 \
    --target-mode random \
    --clear-path-each-episode true
```

**Headless (fast, no GUI):**

```bash
python Evaluation/run_start_to_random_target_pybullet.py --gui false
```

**Fixed target (no random sampling):**

```bash
python Evaluation/run_start_to_random_target_pybullet.py \
    --target 0.030 -0.535 0.110 \
    --target-mode fixed
```

### Environment Modes

The script respects `config.environment.mode`:

- **Default mode** — The obstacle is **hidden** in the PyBullet scene and no obstacle collision is used. Use this for free-space navigation sanity checks.
- **Collision-Free mode** — The obstacle is **visible** in the PyBullet scene (red transparent box) and geometry-based obstacle collision is active. This mirrors the `CartesianPathPlanningEnv` `env_mode="Collision-Free"` behaviour.

Override the mode from the command line:

```bash
# Default mode (no obstacle)
python Evaluation/run_start_to_random_target_pybullet.py --mode Default --gui false

# Collision-Free mode (obstacle visible)
python Evaluation/run_start_to_random_target_pybullet.py --mode Collision-Free --gui false
```

This also applies to `check_real_setup_pybullet.py`:

```bash
python Evaluation/check_real_setup_pybullet.py --mode Default
python Evaluation/check_real_setup_pybullet.py --mode Collision-Free
```

### What is drawn

| Element | Description |
|---------|-------------|
| base_link | Large RGB axes at world origin [0, 0, 0] (axis length 0.10 m) |
| Start sphere | Green solid sphere at the episode start position (radius 0.01 m) |
| Start frame | Small RGB axes at the episode start position (axis length 0.035 m) |
| Target sphere | Yellow solid sphere at the current target position (radius 0.01 m, per episode) |
| Target frame | Small RGB axes at the current target position (axis length 0.035 m) |
| Target region | Yellow wireframe box |
| Workspace | Gray wireframe box |
| Obstacle | Red transparent box with white AABB wireframe, blue centre marker, and info label showing center and size |
| Obstacle info label | "OBSTACLE\ncenter: [...]\nsize: [...]" |
| Black block | Black solid box |
| Plane | Gray transparent plane (below workspace) |
| Path | Coloured line strip (one colour per episode) |

### Log output

Per-episode header:
```
=== Episode 001/100 ===
  start_pos               : [+0.3500, -0.3300, +0.0600]
  target_pos             : [+0.2094, -0.2678, +0.3991]
  straight_line_distance : 0.3592
```

Each step prints:
```
episode=001/100  step=0001  current=[+0.3510,-0.3300,+0.0600]  target=[+0.2094,-0.2678,+0.3991]  dist=0.3498  segment=0.0100  path_length=0.0100
episode=001/100  step=0002  current=[+0.3520,-0.3300,+0.0600]  target=[+0.2094,-0.2678,+0.3991]  dist=0.3399  segment=0.0100  path_length=0.0200
```

Episode summary:
```
--- Episode 001/100 Summary ---
  success                : True
  target_pos            : [+0.2094, -0.2678, +0.3991]
  final_pos             : [+0.2094, -0.2678, +0.3991]
  final_distance        : 0.000000
  straight_line_distance: 0.3592
  path_length           : 0.3592
  num_steps             : 36
```

Final summary (after all episodes):
```
=== Final Summary ===
  episodes             : 100
  success_count       : 100
  success_rate        : 100.0%
  mean_final_distance : 0.000000
  max_final_distance  : 0.000000
  min_final_distance  : 0.000000
  mean_path_length    : 0.3650
  mean_steps          : 36.5
```

---

## Environment Objects

All environment parameters are defined in `config/environment.yaml` — the single source of truth for training, evaluation, and visualization scripts.

### Workspace

The workspace / search region defines the Cartesian bounding box that the agent navigates within.

```
workspace:
  name: search_region
  min: [-0.200, -0.700, 0.020]   # metres
  max: [0.500,  0.000, 0.320]
```

### Target Region

Targets are sampled uniformly from the target region (when `random_target: true`).

```
target_region:
  enabled: true
  random_target: true
  min: [-0.135, -0.700, 0.060]
  max: [ 0.195, -0.370, 0.160]
  fixed_target: [0.030, -0.535, 0.110]
```

### Small Obstacle

The **small obstacle** is the active collision object used in `Collision-Free` mode for geometry-based training collision detection.

```
obstacle:
  enabled: true
  random: false
  name: small_obstacle
  type: box
  center: [0.310, -0.550, 0.080]   # metres
  size:   [0.100, 0.100, 0.100]   # 100 x 100 x 100 mm
  random_region:
    enabled: false
    min: [-0.100, -0.650, 0.060]
    max: [ 0.400, -0.400, 0.160]
```

- `half_extent` is computed in code as `size / 2` = `[0.050, 0.050, 0.050]`
- Obstacle is included in the **15-D observation** (indices 9–14) in `Collision-Free` mode
- The obstacle can be randomised (future feature) by enabling `random_region`

### Plane

The plane is a flat visual surface drawn below the workspace. It is **visual-only** (`collision: false`).

```
plane:
  enabled: true
  z: -0.330        # 330 mm below origin
  center: [0.150, -0.350, -0.330]
  size: [1.0, 1.0]
  color: [0.82, 0.82, 0.82, 0.35]
  collision: false
```

### Fixed Objects

Fixed objects are static visual context (e.g. table base, black block). They are **visual-only** unless `collision: true` is explicitly set.

The **black block** represents the solid table base:

```
fixed_objects:
  - enabled: true
    name: black_block
    type: box
    center: [0.030, -0.550, -0.150]   # metres
    size:   [0.330, 0.330, 0.360]   # 330 x 330 x 360 mm
    color: [0.0, 0.0, 0.0, 1.0]   # solid black
    collision: false
```

- `half_extent` computed in code as `size / 2` = `[0.165, 0.165, 0.180]`
- black_block top surface `z_top = center_z + size_z/2 = -0.150 + 0.180 = 0.030 m`
- `collision: false` — does **not** affect training collision
- Fixed objects are **not** included in the 15-D observation

### PyBullet Visualization

All objects are drawn in PyBullet for visual verification. PyBullet remains **visualization-only** — no URDF, IK/FK, joints, or physics.

### Training Collision

Collision during training is **geometry-based** (AABB slab test in `geometry.py`). PyBullet is never used for collision detection.

| Object | Collision during training? |
|--------|--------------------------|
| Small obstacle (`Collision-Free` mode) | Yes — terminates episode |
| Black block | No (`collision: false`) |
| Plane | No (`collision: false`) |
| PyBullet | Never used for collision |

---

## Environment Design

### CartesianPathPlanningEnv

The environment wraps a simple point-agent in a 3-D Cartesian workspace.

**Action Space**: `Box(-1, 1, shape=(3,), dtype=np.float32)`

| Component | Meaning |
|-----------|---------|
| `ax` | Normalized delta along X-axis |
| `ay` | Normalized delta along Y-axis |
| `az` | Normalized delta along Z-axis |

The environment scales the action by `action_step` (default 0.01 m) to produce
an actual Cartesian displacement per step:
```
delta = action * action_step
next_pos = current_pos + delta
```

**Observation Space**: `Box(-inf, inf, shape=(15,), dtype=np.float32)`

| Indices | Field | Description |
|---------|-------|-------------|
| 0-2 | `current_x/y/z` | Current point position (world frame, m) |
| 3-5 | `target_x/y/z` | Target point position (world frame, m) |
| 6-8 | `err_x/y/z` | target - current (m) |
| 9-11 | `rel_obs_x/y/z` | (obstacle_center - current) / workspace_range |
| 12-14 | `obs_size_x/y/z` | obstacle_half_extent / workspace_range |

In **Default** mode, indices 9-14 are always zeros.

**Workspace** (from `config/environment.yaml`):
```
x: [-0.2,  0.6]  m
y: [-0.7,  0.2]  m
z: [ 0.0,  0.6]  m
```

**Environment Modes**:
- `Default`: No obstacle, free-space navigation.
- `Collision-Free`: A static box obstacle is present. The agent must navigate around it.

---

## Reward Function

```
reward  =  progress_scale * (prev_dist - new_dist)
         - distance_scale  * new_dist
         - step_penalty
         - path_length_scale * path_length

if success:           reward += success_bonus        # +100.0
if obstacle_collision: reward -= collision_penalty     # -100.0
if out_of_workspace:   reward -= workspace_penalty    # -100.0
```

**Default reward config** (`config/environment.yaml`):
| Parameter | Value |
|-----------|-------|
| progress_scale | 10.0 |
| distance_scale | 1.0 |
| step_penalty | 0.01 |
| path_length_scale | 0.1 |
| success_bonus | 100.0 |
| collision_penalty | 100.0 |
| workspace_penalty | 100.0 |
| target_threshold | 0.01 m |
| action_step | 0.01 m |
| max_steps | 200 |

---

## Termination Conditions

| Flag | Trigger |
|------|---------|
| `terminated = True` | Point reached the target (distance < 0.01 m) |
| `truncated = True` | Obstacle collision, workspace violation, or max_steps reached |

---

## Training

### Quick Start

```bash
# DDPG training (500k steps)
python Training/train_ddpg.py --config config/environment.yaml

# SAC training
python Training/train_sac.py --config config/environment.yaml

# TD3 training
python Training/train_td3.py --config config/environment.yaml

# All three algorithms in sequence
python Training/train_all.py --config config/environment.yaml

# With a custom config
python Training/train_ddpg.py --config config/my_setup.yaml
```

### Output Directory

Each training run produces a timestamped output directory:

```
Data/Training/Environment_{MODE}/{ALGORITHM}/FRAME_ONLY/run_{TIMESTAMP}/
├── config.json              # Experiment configuration
├── model/
│   ├── final_model.zip     # Saved policy
│   └── best_model.zip      # Best model encountered
├── logs/
│   ├── progress.csv         # SB3 training metrics
│   ├── monitor.csv          # Episode rewards and lengths
│   └── time.txt             # Elapsed training time
├── tensorboard/             # Optional TensorBoard logs
└── vec_normalize_stats.pkl # (legacy — not created in raw-observation pipeline)
```

### Headless Training

Set `ENABLE_GUI` to `False` (not applicable to this project since there is no GUI).
The training scripts automatically use CPU/CUDA based on availability:

```python
# In each training script:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### TensorBoard

```bash
tensorboard --logdir Data/Training/
```

### Multi-Environment Parallel Training

Training can use multiple parallel CPU environments to increase training throughput.
By default, a single environment is used. Pass `--n-envs` to enable parallelism:

- `--n-envs 1` (default): Uses `DummyVecEnv` — no subprocess overhead.
- `--n-envs N` (N > 1): Uses `SubprocVecEnv` with `fork` — true multiprocess.

```bash
# Benchmark — single env (DummyVecEnv)
python Training/train_ddpg.py \
  --config config/environment.yaml \
  --algo-config config/experiments/ddpg_default.yaml \
  --total-timesteps 50000 \
  --n-envs 1

# Benchmark — 4 parallel envs (SubprocVecEnv, fork)
python Training/train_ddpg.py \
  --config config/environment.yaml \
  --algo-config config/experiments/ddpg_default.yaml \
  --total-timesteps 50000 \
  --n-envs 4
```

**Expected results:** SubprocVecEnv should yield higher FPS for CPU-bound workloads. Compare the wall-clock time printed at the end of each run, or check `logs/time.txt` in the run directory.

Configuration options (set in `config/experiments/ddpg_default.yaml` or via CLI):

| Option | Default | Description |
|--------|---------|-------------|
| `n_envs` | 1 | Number of parallel environments |
| `vec_env_type` | auto | `auto`, `dummy`, or `subproc` |
| `vec_normalize` | false | VecNormalize disabled (raw observation pipeline) |
| `norm_obs` | false | (disabled) |
| `norm_reward` | false | (disabled) |
| `progress_bar` | false | Show tqdm progress bar |
| `log_interval` | 50 | Print frequency (episodes) |

---

## Evaluation

### Static Target Prediction

```bash
python Evaluation/predict_static.py \
    --model Data/Training/.../final_model.zip \
    --start 0.0 -0.3 0.2 \
    --target 0.4 -0.4 0.5
```

Output: `Data/Prediction/Environment_Default/{ALGORITHM}/FRAME_ONLY/trajectory_static_target.csv`

### Random Target Evaluation

```bash
python Evaluation/predict_random.py \
    --model Data/Training/.../final_model.zip \
    --algorithm DDPG \
    --num-episodes 100
```

Output: `metrics_random_N_100.csv` with columns: episode, is_success, cumulative_reward, episode_length, final_distance.

### Quantitative Evaluation

```bash
python Evaluation/evaluate_model.py \
    --model Data/Training/.../final_model.zip \
    --num-episodes 50
```

---

## Exporting Waypoints

The waypoint export script generates a clean waypoint list for downstream
ROS2 / MoveIt consumption:

```bash
python Evaluation/export_waypoints.py \
    --model Data/Training/.../final_model.zip \
    --algorithm DDPG \
    --start 0.0 -0.3 0.2 \
    --target 0.4 -0.4 0.5 \
    --output Data/Waypoints/
```

**Waypoint CSV format**:

| Column | Description |
|--------|-------------|
| `step` | Step number |
| `x`, `y`, `z` | Waypoint position (m) |
| `distance_to_target` | Euclidean distance to target (m) |
| `action_x/y/z` | Normalized action taken to reach this waypoint |
| `reward` | Reward received for this step |
| `is_success` | Whether the target was reached |
| `is_collision` | Whether a collision occurred |
| `is_out_of_workspace` | Whether the workspace was violated |

**Waypoint JSON format** additionally includes metadata:
```json
{
  "metadata": {
    "algorithm": "DDPG",
    "env_mode": "Default",
    "start_pos": [0.0, -0.3, 0.2],
    "target_pos": [0.4, -0.4, 0.5],
    "total_steps": 142,
    "final_success": true,
    "final_distance": 0.008
  },
  "waypoints": [...]
}
```

---

## Integration with the MoveIt Layer

The waypoint list produced by this project is designed to be consumed by a
separate ROS2 / MoveIt execution layer (Layer 2). The integration path:

1. **Train** the DRL policy in this project (Layer 1).
2. **Export** the waypoint list from a trained model using `export_waypoints.py`.
3. **Transfer** the waypoint CSV/JSON to the robot's execution machine.
4. **Layer 2** (ROS2 / MoveIt) receives the waypoints and:
   - Converts Cartesian waypoints to joint trajectories via IK
   - Validates each waypoint against the robot's collision model
   - Checks joint limits and singularity conditions
   - Generates a time-parameterized trajectory with velocity/acceleration profiles
   - Executes the trajectory on the physical robot

The waypoint list format is intentionally simple: `step, x, y, z, distance`.
This allows easy parsing and processing in the Layer 2 ROS2 node without
requiring knowledge of the DRL training implementation.

---

## Troubleshooting

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())   # Should be True with CUDA wheels
```

If False, reinstall PyTorch with the CUDA index URL:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### PyBullet GUI on Linux / Headless Servers

PyBullet GUI requires an OpenGL context. On headless servers, use:

```bash
# Option 1: use --gui false to run headless (no visualisation)
python Evaluation/test_pybullet_env.py --gui false

# Option 2: enable OS-level offscreen rendering
python -c "import pybullet as p; p.connect(p.DIRECT)"
```

On Linux, install EGL for hardware-accelerated rendering:
```bash
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev
```

### PyBullet Not Installed

If PyBullet is not installed, the environment runs headless without error:

```bash
pip install pybullet
```

To check PyBullet availability:
```bash
python scripts/check_install.py
```

### Module Import Errors

Ensure you run from the project root directory, or that `src/` is in `PYTHONPATH`.
The scripts automatically add `src/` to `sys.path` relative to their own location.

### Training Convergence

If the agent does not converge:
- Increase `total_timesteps` (500k may be too few for complex tasks).
- Adjust the reward function weights in `config/environment.yaml`.
- Try a different algorithm (SAC often converges more reliably than DDPG).
- Increase the neural network size with `policy_kwargs: {net_arch: [512, 512, 512]}`.

### Large Action Steps Causing Oscillation

If the agent oscillates around the target:
- Reduce `action_step` in `config/environment.yaml` (e.g., from 0.01 to 0.005).
- Increase the `success_bonus` to encourage reaching the target quickly.

---

## Future Improvements

- **Hindsight Experience Replay (HER)**: Use SB3's `HerReplayBuffer` to improve
  sample efficiency for sparse-reward tasks.
- **Dynamic obstacle avoidance**: Add moving or multiple obstacles.
- **Curriculum learning**: Gradually increase obstacle complexity during training.
- **Multi-target waypoint sequences**: Train on sequences of waypoints instead
  of single start-target pairs.
- **Layer 2 ROS2 package**: Implement the MoveIt integration layer that receives
  the waypoint list and executes on the physical robot.

---

## License

MIT License
