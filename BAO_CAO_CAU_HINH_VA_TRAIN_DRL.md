# Bao cao cau hinh tham so va quy trinh train hai project DRL

Tai lieu nay tom tat cach cau hinh tham so, cach train, train tiep va kiem tra ket qua cho hai project:

- `DRL_Pathplanning_trainning`
- `DRL_Robot_Manipulator`

Noi dung duoc tong hop truc tiep tu cac file source, config va script training hien co trong repository.

## 1. Chuan bi moi truong chung

Thu muc goc repository:

```powershell
cd C:\Users\MinhQuang\DRL
```

Nen dung mot virtual environment rieng cho tung project hoac dung `.venv` o root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Co the cai dependency chung tu root:

```powershell
pip install -r requirements.txt
```

Hoac cai rieng cho tung project:

```powershell
pip install -r DRL_Pathplanning_trainning\requirements.txt
pip install -r DRL_Robot_Manipulator\requirements.txt
```

Luu y ve PyTorch CUDA:

- `DRL_Pathplanning_trainning\requirements.txt` pin `torch==2.2.2+cu121`, can dung PyTorch CUDA index neu cai ban GPU:

```powershell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r DRL_Pathplanning_trainning\requirements.txt
```

- Neu may khong co GPU, dung ban CPU va sua requirement neu can.
- Cac script train robot manipulator dang hard-code `device='cuda'` trong model, nen can doi sang `device='cpu'` neu may khong co CUDA.

Kiem tra nhanh package:

```powershell
python -c "import gymnasium, stable_baselines3, pybullet, torch; print('OK', torch.cuda.is_available())"
```

## 2. Project DRL_Pathplanning_trainning

### 2.1 Muc tieu va mo hinh moi truong

Project nay train point-agent 3D trong khong gian Cartesian. Agent khong load robot, khong IK/FK, khong MoveIt. Action la vector delta 3 chieu, observation la vector 15 chieu.

File moi truong chinh:

- `DRL_Pathplanning_trainning\src\drl_pathplanning\gymnasium\cartesian_frame_env.py`
- `DRL_Pathplanning_trainning\src\drl_pathplanning\gymnasium\config.py`
- `DRL_Pathplanning_trainning\src\drl_pathplanning\gymnasium\reward.py`
- `DRL_Pathplanning_trainning\src\drl_pathplanning\training\trainer.py`

Action space:

```text
Box(-1, 1, shape=(3,))
delta = action * environment.action_step
next_pos = current_pos + delta
```

Observation 15 chieu:

```text
[current_xyz, target_xyz, error_xyz, relative_obstacle_xyz, obstacle_size_xyz]
```

Neu `obstacle.enabled=false`, 6 gia tri cuoi ve obstacle se la 0.

### 2.2 File cau hinh chinh

File cau hinh dang duoc dung:

```text
DRL_Pathplanning_trainning\config\environment.yaml
```

File `environment_yaskawa.yaml` la bien the khac. Cac file trong `config\deprecated\` la cau hinh cu, khong nen dung lam nguon chinh neu khong co ly do.

Cac script train doc cung mot file YAML hop nhat. Script nao duoc chay se tu override algorithm:

- `Training\train_td3.py` set `cfg.training.algorithm = "TD3"`
- `Training\train_ddpg.py` set `cfg.training.algorithm = "DDPG"`
- `Training\train_sac.py` set `cfg.training.algorithm = "SAC"`
- `Training\train_ppo.py` set `cfg.training.algorithm = "PPO"`

### 2.3 Cac nhom tham so can chinh

#### Environment

Trong `environment.yaml`:

```yaml
environment:
  observation_type: frame_only
  action_step: 0.01
  max_episode_steps: 300
```

Y nghia:

- `action_step`: do dai buoc di thuc te tinh bang met khi action = 1. Giam xuong neu agent dao dong quanh target, tang len neu can di nhanh hon.
- `max_episode_steps`: so lan `step()` toi da moi episode.
- `observation_type`: hien tai chi ho tro `frame_only`.

#### Workspace

```yaml
workspace:
  x_min: 0.4250
  x_max: 0.675
  y_min: -0.200
  y_max: 0.200
  z_min: 0.02
  z_max: 0.600
```

Day la hop gioi han tim kiem. Start, target va obstacle phai nam trong workspace, neu khong loader se bao loi validate.

#### Start va target

```yaml
start:
  mode: random
  fixed_position: [0.55, 0.00, 0.30]
  random_bounds:
    min: [0.4250, -0.200, 0.02]
    max: [0.675, 0.200, 0.600]

target:
  mode: random
  fixed_position: [0.55, 0.00, 0.10]
  random_bounds:
    min: [0.4250, -0.200, 0.02]
    max: [0.675, 0.200, 0.10]
```

Y nghia:

- `mode: fixed`: dung `fixed_position`.
- `mode: random`: sample ngau nhien trong `random_bounds`.
- Neu train policy tong quat, nen dung random.
- Neu debug mot ca cu the, dung fixed de de lap lai.

#### Obstacle va collision

```yaml
obstacle:
  enabled: false
  mode: random
  center: [0.1450, -0.550, 0.080]
  size: [0.100, 0.100, 0.100]
  safety_margin: 0.03

collision:
  enabled: true
```

Y nghia:

- `obstacle.enabled=false`: train free-space, obstacle khong xuat hien trong observation.
- `obstacle.enabled=true` va `collision.enabled=true`: obstacle duoc dua vao observation va collision checker.
- `obstacle.size` la kich thuoc day du cua box, code tu tinh half extent bang `size / 2`.
- `safety_margin` duoc cong vao vung collision.
- `obstacle.mode=random`: moi episode sample obstacle center theo `obstacle.random_bounds`.
- `obstacle.size_random.enabled=true`: random kich thuoc obstacle theo cac min/max.

#### Termination

```yaml
termination:
  goal_threshold: 0.03
  collision_terminate: true
  workspace_terminate: true
```

Y nghia:

- Success khi khoang cach den target nho hon `goal_threshold`.
- Collision hoac vuot workspace co the ket thuc episode neu flag tuong ung la `true`.

#### Reward

Reward hien tai la simple distance-based:

```text
reward = r_success + r_collision + r_distance + r_workspace + r_episode + r_time + r_shake
```

Trong YAML:

```yaml
reward:
  success_bonus: 20.0
  collision_penalty: 300.0
  workspace_penalty: 300.0
  timeout_penalty: 50.0
  distance_scale: 1.5
  time_penalty: 0.0
  shake_penalty_scale: 0.005
  shake_window: 10
  shake_dot_threshold: 0.0
  shake_min_movement: 1.0e-6
```

Goi y chinh:

- Agent khong toi target: tang `success_bonus`, tang `distance_scale`, hoac giam `action_step`.
- Agent hay va cham: tang `collision_penalty`, tang `safety_margin`, hoac bat obstacle curriculum.
- Agent di rung/doi huong lien tuc: tang `shake_penalty_scale`.
- Episode ket thuc vi timeout qua nhieu: tang `max_episode_steps` hoac tang `action_step`.

#### Visualization

```yaml
visualization:
  enabled: true
  gui: true
  show_workspace: true
  show_target_region: true
  show_table: true
  show_path: true
```

Luu y:

- Training mac dinh chay headless vi cac script train co CLI `--gui false`.
- Chi bat GUI cho debug ngan.
- Neu `--gui true` va `training.n_envs > 1`, trainer se force ve `n_envs=1`.

#### Training chung

```yaml
training:
  total_timesteps: 10000000
  seed: 42
  device: cuda
  n_envs: 30
  vec_env_type: auto
  progress_bar: true
  log_interval: 50
  episode_log_interval: 10
  eval_freq: 50000
  save_freq: 25000
```

Y nghia:

- `total_timesteps`: so buoc train mac dinh.
- `n_envs`: so environment song song. `auto` dung `DummyVecEnv` khi `n_envs=1`, dung `SubprocVecEnv` khi `n_envs>1`.
- `eval_freq`: tan suat evaluate va luu `best_model.zip`.
- `save_freq`: tan suat luu checkpoint `checkpoint_t{step}.zip`.
- `device` trong YAML duoc luu vao snapshot, nhung trainer chung hien tu chon device bang `cuda` neu `torch.cuda.is_available()`, nguoc lai `cpu`.

#### Hyperparameter theo algorithm

TD3:

```yaml
td3:
  learning_rate: 0.0003
  buffer_size: 1000000
  learning_starts: 10000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  policy_delay: 2
  target_policy_noise: 0.2
  target_noise_clip: 0.5
  policy_kwargs:
    net_arch: [256, 256]
```

DDPG:

```yaml
ddpg:
  learning_rate: 0.0003
  buffer_size: 1000000
  learning_starts: 10000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  policy_kwargs:
    net_arch: [256, 256]
```

SAC:

```yaml
sac:
  learning_rate: 0.0003
  buffer_size: 1000000
  learning_starts: 10000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  ent_coef: auto
  policy_kwargs:
    net_arch: [256, 256]
```

PPO:

```yaml
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  policy_kwargs:
    net_arch: [256, 256]
```

Action noise:

```yaml
action_noise:
  enabled: true
  type: NormalActionNoise
  mean: 0.0
  sigma: 0.1
```

Chi TD3 va DDPG dung `action_noise`. SAC va PPO bo qua phan nay.

### 2.4 Cach train tu dau

Chay tu thu muc project:

```powershell
cd C:\Users\MinhQuang\DRL\DRL_Pathplanning_trainning
..\.venv\Scripts\Activate.ps1
```

Neu dung `.venv` o root, lenh activate dung:

```powershell
..\.venv\Scripts\Activate.ps1
```

Neu PowerShell khong chap nhan do khoang trang, dung duong dan ro:

```powershell
C:\Users\MinhQuang\DRL\.venv\Scripts\Activate.ps1
```

Train TD3:

```powershell
python Training\train_td3.py --config config\environment.yaml
```

Train DDPG:

```powershell
python Training\train_ddpg.py --config config\environment.yaml
```

Train SAC:

```powershell
python Training\train_sac.py --config config\environment.yaml
```

Train PPO:

```powershell
python Training\train_ppo.py --config config\environment.yaml
```

Override so buoc train nhanh tu CLI:

```powershell
python Training\train_sac.py --config config\environment.yaml --timesteps 50000
```

Bat GUI de debug:

```powershell
python Training\train_td3.py --config config\environment.yaml --timesteps 5000 --gui true --show true --render-sleep 0.02 --render-first-episodes 3
```

Khuyen nghi:

- Train dai: `--gui false`, `progress_bar true`, `n_envs` tuy CPU.
- Debug hinh hoc: `--gui true`, `n_envs=1`.
- Windows voi `n_envs>1` dung spawn, co the cham hon Linux. Neu gap loi multiprocessing, dat `training.n_envs: 1`.

### 2.5 Output train

Output duoc tao theo mau:

```text
DRL_Pathplanning_trainning\Data\Training\{ALGORITHM}\FRAME_ONLY\run_{YYYYMMDD_HHMMSS}\
  config.yaml
  config.json
  run_info.json
  model\
    final_model.zip
    best_model.zip
    best_model_info.json
    checkpoint_t*.zip
  logs\
    progress.csv
    monitor.csv
    time.txt
  tensorboard\
  trajectory\
  replay_buffer.pkl
```

Trong do:

- `final_model.zip`: model cuoi cung.
- `best_model.zip`: model tot nhat theo evaluation callback.
- `checkpoint_t*.zip`: checkpoint dinh ky theo `training.save_freq`.
- `replay_buffer.pkl`: chi co y nghia voi TD3/DDPG/SAC.
- `config.yaml` la ban copy raw config dung cho run, rat quan trong de tai lap thi nghiem.

Xem TensorBoard:

```powershell
tensorboard --logdir DRL_Pathplanning_trainning\Data\Training
```

### 2.6 Train tiep tu checkpoint/model

TD3 script chinh ho tro load model va replay buffer:

```powershell
python Training\train_td3.py `
  --config config\environment.yaml `
  --load-model Data\Training\TD3\FRAME_ONLY\run_xxx\model\final_model.zip `
  --load-replay-buffer Data\Training\TD3\FRAME_ONLY\run_xxx\replay_buffer.pkl `
  --timesteps 1000000
```

Continue script rieng cho TD3, DDPG, SAC, PPO:

```powershell
python Training\train_td3_continue_checkpoint.py --checkpoint Data\Training\TD3\FRAME_ONLY\run_xxx\model\checkpoint_t250000.zip --timesteps 1000000
python Training\train_ddpg_continue_checkpoint.py --checkpoint Data\Training\DDPG\FRAME_ONLY\run_xxx\model\checkpoint_t250000.zip --timesteps 1000000
python Training\train_sac_continue_checkpoint.py --checkpoint Data\Training\SAC\FRAME_ONLY\run_xxx\model\checkpoint_t250000.zip --timesteps 1000000
python Training\train_ppo_continue_checkpoint.py --checkpoint Data\Training\PPO\FRAME_ONLY\run_xxx\model\checkpoint_t250000.zip --timesteps 1000000
```

Mac dinh cac continue script off-policy khong load replay buffer cu, ma tao replay buffer moi. Dieu nay phu hop khi checkpoint tot hon final model nhung buffer cu khong con phu hop.

### 2.7 Evaluation va debug

Predict model SAC:

```powershell
python Evaluation\predict_model_sac.py --episodes 10 --gui false --show true
```

Chi dinh run cu the:

```powershell
python Evaluation\predict_model_sac.py --run Data\Training\SAC\FRAME_ONLY\run_20260608_131757
```

Tuong tu co cac file:

- `Evaluation\predict_model_ddpg.py`
- `Evaluation\predict_model_sac.py`
- `Evaluation\predict_model_ppo.py`
- `Evaluation\predict_model.py`

Tim checkpoint tot:

```powershell
python Evaluation\find_best_checkpoint.py --run Data\Training\TD3\FRAME_ONLY\run_xxx
```

Plot ket qua:

```powershell
python Evaluation\plot_results.py --run Data\Training\TD3\FRAME_ONLY\run_xxx
```

Debug moi truong start-to-target:

```powershell
python Evaluation\test_environment_start_to_target.py --config config\environment.yaml --episodes 10 --gui true
```

## 3. Project DRL_Robot_Manipulator

### 3.1 Muc tieu va mo hinh moi truong

Project nay train robot manipulator Yaskawa GP7 trong PyBullet. Agent dieu khien TCP bang delta Cartesian 3 chieu. Moi step:

1. Scale action bang `action_step`.
2. Tinh desired TCP position.
3. Kiem tra workspace.
4. Giai IK bang PyBullet/RoLE helper.
5. Set joint target trong PyBullet.
6. Doc TCP pose that tu PyBullet.
7. Tinh reward, success, collision va truncation.

File moi truong dang dung cho train:

```text
DRL_Robot_Manipulator\src\Industrial_Robotics_Gym\Environment\GP7ReachPyBulletEnv.py
```

File registration Gym:

```text
DRL_Robot_Manipulator\src\Industrial_Robotics_Gym\__init__.py
```

Environment ID dang duoc register:

```text
YaskawaGP7ReachPyBullet-Default-v0
YaskawaGP7ReachPyBullet-Collision-Free-v0
```

### 3.2 File cau hinh trong project robot

File `DRL_Robot_Manipulator\config\config.yaml` hien chi co:

```yaml
PROJECT_FOLDER_NAME: '_Robot_Manipulator'
```

File nay duoc `src\config_loader.py` dung de tim project root. No khong chua hyperparameter training.

Cac tham so thuc su nam trong:

- `Training\train_ddpg_gp7.py`, `Training\train_ddpg.py`, `Training\train_sac.py`, `Training\train_td3.py`
- `Training\continue_train_ddpg_gp7.py`
- `src\Industrial_Robotics_Gym\Environment\GP7ReachPyBulletEnv.py`
- `src\PyBullet\Configuration\Environment.py`
- `src\RoLE\Parameters\Robot.py`

### 3.3 Tham so moi truong GP7ReachPyBulletEnv

Constructor trong `GP7ReachPyBulletEnv.py`:

```python
GP7ReachPyBulletEnv(
    enable_gui=True,
    action_step=0.01,
    distance_thresh=0.03,
    max_episode_steps=200,
    env_mode="Default",
)
```

Y nghia:

- `enable_gui`: bat/tat PyBullet GUI.
- `action_step`: met moi step khi action = 1.
- `distance_thresh`: nguong success.
- `max_episode_steps`: so step toi da.
- `env_mode`: `Default` hoac `Collision-Free`.

Trong cac script train hien tai, `gym.make(CONST_ENV_ID, enable_gui=ENABLE_GUI)` chi truyen `enable_gui`, nen cac gia tri `action_step=0.01`, `distance_thresh=0.03`, `max_episode_steps=200` duoc lay theo default cua environment.

Neu muon override trong training script, sua dong tao env thanh:

```python
raw_env = gym.make(
    CONST_ENV_ID,
    enable_gui=ENABLE_GUI,
    action_step=0.01,
    distance_thresh=0.03,
    max_episode_steps=200,
)
```

Action space:

```text
Box(-1, 1, shape=(3,))
delta = action * action_step
```

Observation 15 chieu:

```text
[tcp_xyz, target_xyz, error_xyz, relative_obstacle_xyz, obstacle_size_xyz]
```

Reward:

- Default: `reward = -distance(tcp, target)`
- Collision-Free: `reward = -(distance + soft_collision_penalty)`
- Collision that: reward `-5.0`, episode truncated.
- Workspace/IK/joint-limit failure: reward `-1.0`, episode truncated.
- Hien khong co success bonus.

Termination:

- `terminated=True`: TCP cach target nho hon `distance_thresh`.
- `truncated=True`: workspace violation, IK fail, joint limit, collision, hoac qua `max_episode_steps`.

IK setting trong environment:

```python
__ik_props = {
    "delta_time": 0.01,
    "num_of_iteration": 500,
    "tolerance": 1e-30,
    "use_orientation": True,
    "ik_position_tolerance": 0.01,
}
```

Luu y: mot so `config.json` trong train script ghi note `use_orientation=False`, nhung source environment hien tai dang dat `use_orientation=True`. Khi can doi IK, chinh trong `GP7ReachPyBulletEnv.py` moi la nguon co tac dung.

### 3.4 Cau hinh workspace, target, obstacle

File:

```text
DRL_Robot_Manipulator\src\PyBullet\Configuration\Environment.py
```

YASKAWA GP7 Default:

```python
Search center = [0.150, -0.350, 0.500]
Search size   = [0.700,  0.700, 0.300]
Target center = [0.030, -0.505, 0.410]
Target size   = [0.330,  0.330, 0.100]
Collision_Object = None
```

YASKAWA GP7 Collision-Free:

```python
Search center = [0.150, -0.350, 0.500]
Search size   = [0.700,  0.700, 0.300]
Target center = [0.030, -0.550, 0.470]
Target size   = [0.330,  0.330, 0.010]
Obstacle pos  = [0.130, -0.500, 0.410]
Obstacle type = cube100
```

Scene object `Table` duoc load lam vat canh/ban, nhung `Enable_Collision=False` trong scene object. Learning obstacle chinh la `Collision_Object`.

ARM cung co cau hinh Search/Target/Collision trong file nay, nhung training environment hien tai `GP7ReachPyBulletEnv` dang hard-code `YASKAWA_GP7_Str`. Neu muon train ARM, can tao environment rieng hoac refactor env de nhan robot structure/URDF theo tham so.

### 3.5 Cau hinh robot va joint limit

File:

```text
DRL_Robot_Manipulator\src\RoLE\Parameters\Robot.py
```

YASKAWA:

- Robot name: `YASKAWA_GP7`
- URDF path: `URDFs\Robots\YASKAWA_GP7\YASKAWA_GP7.urdf`
- TCP link name: default `ee_link`
- Joint limits nam trong `YASKAWA_GP7_Str.Theta.Limit`
- Home pose nam trong `YASKAWA_GP7_Str.Theta.Home`

ARM:

- Robot name: `ARM`
- URDF path: `URDFs\Robots\ARM\ARM.urdf`
- TCP link name: `tcp_link`
- Co 6 revolute joints va 2 prismatic gripper joints.

### 3.6 Cach chinh tham so training robot manipulator

Cac script training hien tai dung hang so o dau file.

`Training\train_ddpg_gp7.py`:

```python
CONST_ENV_MODE = "Default"
CONST_ALGORITHM = "DDPG"
ENABLE_GUI = True
CONST_TOTAL_TIMESTEPS = 500000
CONST_LOG_INTERVAL = 10
SEED = 42
```

Model DDPG:

```python
gamma=0.95
learning_rate=0.001
batch_size=256
policy_kwargs=dict(net_arch=[256, 256, 256])
action_noise sigma=0.1
device="cuda"
```

`Training\train_ddpg.py`:

```python
CONST_ENV_MODE = "Collision-Free"
CONST_ALGORITHM = "DDPG"
ENABLE_GUI = True
CONST_TOTAL_TIMESTEPS = 100000
```

`Training\train_sac.py`:

```python
CONST_ENV_MODE = "Collision-Free"
CONST_ALGORITHM = "SAC"
ENABLE_GUI = True
CONST_TOTAL_TIMESTEPS = 100000
```

SAC khong dung action noise.

`Training\train_td3.py`:

```python
CONST_ENV_MODE = "Collision-Free"
CONST_ALGORITHM = "TD3"
ENABLE_GUI = True
CONST_TOTAL_TIMESTEPS = 100000
```

TD3 dung `NormalActionNoise` sigma `0.1`.

Khuyen nghi khi train dai:

- Doi `ENABLE_GUI = False`.
- Neu khong co GPU, doi `device='cuda'` thanh `device='cpu'` hoac `device='auto'`.
- Tang `CONST_TOTAL_TIMESTEPS` cho thi nghiem that.
- Giam `learning_rate` neu reward dao dong manh.
- Tang `distance_thresh` de bai toan de hon trong giai do debug, giam lai khi can do chinh xac.

### 3.7 Cach train robot manipulator tu dau

Chay tu thu muc project:

```powershell
cd C:\Users\MinhQuang\DRL\DRL_Robot_Manipulator
C:\Users\MinhQuang\DRL\.venv\Scripts\Activate.ps1
```

Train DDPG Default cho GP7:

```powershell
python Training\train_ddpg_gp7.py
```

Train DDPG Collision-Free:

```powershell
python Training\train_ddpg.py
```

Train SAC Collision-Free:

```powershell
python Training\train_sac.py
```

Train TD3 Collision-Free:

```powershell
python Training\train_td3.py
```

Neu muon chay headless, sua trong script:

```python
ENABLE_GUI = False
```

Neu muon chay CPU, sua dong tao model:

```python
device="cpu"
```

### 3.8 Output train robot manipulator

Output moi run:

```text
DRL_Robot_Manipulator\Data\Training\Environment_{MODE}\{ALGORITHM}\YASKAWA_GP7\run_{YYYYMMDD_HHMMSS}\
  config.json
  logs\
    progress.csv
    monitor.csv
    time.txt
  model\
    final_model.zip
  tensorboard\
```

`train_ddpg_gp7.py` co them luu replay buffer:

```text
replay_buffer.pkl
```

Xem TensorBoard:

```powershell
tensorboard --logdir DRL_Robot_Manipulator\Data\Training
```

### 3.9 Train tiep DDPG GP7

File:

```text
DRL_Robot_Manipulator\Training\continue_train_ddpg_gp7.py
```

Nen dung CLI thay vi sua hard-code path trong script:

```powershell
python Training\continue_train_ddpg_gp7.py `
  --run-dir Data\Training\Environment_Default\DDPG\YASKAWA_GP7\run_YYYYMMDD_HHMMSS `
  --timesteps 300000 `
  --env-mode Default `
  --device auto
```

Neu co model path truc tiep:

```powershell
python Training\continue_train_ddpg_gp7.py `
  --model-path Data\Training\Environment_Default\DDPG\YASKAWA_GP7\run_YYYYMMDD_HHMMSS\model\final_model.zip `
  --timesteps 300000 `
  --env-mode Default `
  --device auto
```

Load replay buffer neu run cu co `replay_buffer.pkl`:

```powershell
python Training\continue_train_ddpg_gp7.py `
  --run-dir Data\Training\Environment_Default\DDPG\YASKAWA_GP7\run_YYYYMMDD_HHMMSS `
  --timesteps 300000 `
  --load-replay-buffer
```

Script se tao output moi:

```text
Data\Training\Environment_{mode}\DDPG\YASKAWA_GP7\continue_{YYYYMMDD_HHMMSS}\
```

### 3.10 Kiem tra va evaluation robot manipulator

Kiem tra Gym env:

```powershell
python Evaluation\Gym\Environment\check_env.py
```

Chay random action trong GUI:

```powershell
python Evaluation\Gym\Environment\test_env.py
```

Kiem tra configuration space YASKAWA:

```powershell
python Evaluation\PyBullet\Control\test_configuration_space_rand.py
python Evaluation\PyBullet\Control\test_configuration_space_vertices.py
```

Kiem tra configuration space ARM:

```powershell
python Evaluation\PyBullet\Control\test_configuration_space_arm.py
python Evaluation\PyBullet\Control\test_configuration_space_vertices_arm.py
```

Chay nhanh headless cho ARM:

```powershell
python Evaluation\PyBullet\Control\test_configuration_space_arm.py --headless --samples 10 --sleep 0
python Evaluation\PyBullet\Control\test_configuration_space_vertices_arm.py --headless --sleep 0
```

Prediction model nam trong:

```text
Evaluation\Gym\Model\Prediction\Static
Evaluation\Gym\Model\Prediction\Random
```

Training plot/compare nam trong:

```text
Evaluation\Gym\Model\Training
```

## 4. Goi y thuc nghiem

### 4.1 Khi muon train nhanh de kiem tra pipeline

Pathplanning:

```yaml
training:
  total_timesteps: 50000
  n_envs: 1
  progress_bar: true
```

Robot manipulator:

```python
ENABLE_GUI = False
CONST_TOTAL_TIMESTEPS = 10000
```

### 4.2 Khi muon train nghiem tuc

Pathplanning:

- Dung `n_envs` cao neu CPU du manh.
- Luu `config.yaml` cua run.
- Theo doi `success_rate`, `collision_rate`, `workspace_violation_rate`, `reward/total`.
- Dung `best_model.zip` de evaluate, khong chi nhin `final_model.zip`.

Robot manipulator:

- Tat GUI.
- Giu seed trong `config.json`.
- Khong commit output train lon.
- Neu train Collision-Free, quan sat `is_collision`, `contacts`, `termination_reason`.

### 4.3 Khi thay agent hoc kem

Kiem tra theo thu tu:

1. Target va start co nam trong workspace khong.
2. `action_step` co qua lon hoac qua nho khong.
3. `distance_thresh` co qua chat khong.
4. Reward co qua phat collision/workspace khong.
5. Neu robot manipulator, IK co fail nhieu khong va target co nam trong reachable target space khong.
6. Neu collision-free, obstacle co nam chet giua duong duy nhat khong.
7. Device va dependency co dung khong.

## 5. Quy tac commit output

Nen commit:

- Source code `.py`
- Config `.yaml`
- README, report `.md`
- URDF, mesh, texture can de load robot

Khong nen commit:

- Virtual environment `.venv`, `venv`
- `__pycache__`, `.pyc`
- Model `.zip`
- Replay buffer `.pkl`
- TensorBoard event files
- `Data\Training`, `Data\Prediction`, `Data\Model` neu la output lon

Repo da co `.gitignore` cho cac nhom output nay. Neu mot file binary da tracked tu truoc, `.gitignore` se khong tu bo tracked file do.

## 6. Checklist train lai tu dau

1. Activate venv.
2. Cai dependency dung project.
3. Chon project can train.
4. Chinh config:
   - Pathplanning: `config\environment.yaml`
   - Robot manipulator: constants trong `Training\*.py` va env/scene config trong `src\...`
5. Chay debug ngan voi it timestep hoac GUI.
6. Chay train dai headless.
7. Kiem tra output trong `Data\Training`.
8. Evaluate `best_model.zip` va `final_model.zip`.
9. Ghi lai config/run id/model dung cho bao cao.
