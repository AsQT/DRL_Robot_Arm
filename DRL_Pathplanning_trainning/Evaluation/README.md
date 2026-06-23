# Evaluation Scripts

Thu muc `Evaluation/` chua cac script de:

- kiem tra moi truong truoc khi train
- predict/evaluate model da train
- ve bieu do tu log train
- audit observation/reward

Tat ca lenh ben duoi chay trong **VS Code Terminal tren Windows, profile PowerShell**.

## Chay Tu Dau

Dung virtual environment chung o repo root `C:\Users\MinhQuang\DRL\.venv`:

```powershell
cd C:\Users\MinhQuang\DRL
.\.venv\Scripts\Activate.ps1
cd .\DRL_Pathplanning_trainning
```

Sau khi activate, dung `python` truc tiep:

```powershell
python .\Evaluation\<script>.py
```

## Script Nen Dung Truoc Khi Training

### `test_environment_start_to_target.py`

Day la script kiem tra moi truong quan trong nhat truoc khi train. No:

- load `config/environment.yaml`
- tao `CartesianPathPlanningEnv`
- sample start/target giong logic train that
- chay bo dieu khien hinh hoc don gian theo duong thang
- bao ket qua theo episode: success, collision, workspace violation, max steps

Lenh co ban:

```powershell
python .\Evaluation\test_environment_start_to_target.py --config .\config\environment.yaml --episodes 10 --steps 50 --gui true --show false
```

Chay headless nhanh:

```powershell
python .\Evaluation\test_environment_start_to_target.py --config .\config\environment.yaml --episodes 3 --steps 50 --gui false --show false
```

Lenh nay rat hop de tra loi 3 cau hoi:

- workspace hien tai co hop ly khong
- start/target random co nam dung vung khong
- obstacle random co qua gan duong di khong

Mot so flag hay dung:

- `--start-mode config|fixed|random`
- `--start X Y Z`
- `--target X Y Z`
- `--episodes N`
- `--steps N`
- `--gui true|false`
- `--show true|false`
- `--verbose true|false`
- `--debug-obstacle true|false`
- `--debug-body-audit true|false`
- `--debug-aabb true|false`
- `--debug-obs-detail true|false`

Vi du debug obstacle:

```powershell
python .\Evaluation\test_environment_start_to_target.py --config .\config\environment.yaml --episodes 3 --steps 50 --gui false --show false --debug-obstacle true
```

## Predict / Evaluate Model

Co 4 script predict chinh:

- `predict_model.py` cho `TD3`
- `predict_model_ddpg.py`
- `predict_model_sac.py`
- `predict_model_ppo.py`

Tat ca dung chung helper trong `predict_common.py`.

### Luu y quan trong

- `predict_model_sac.py` tu tim `run_*` moi nhat co model hop le.
- `predict_model_ddpg.py` va `predict_model_ppo.py` mac dinh tro toi thu muc algorithm, khong phai mot `run_*` cu the.
- `predict_model.py` cho `TD3` hien con giu default run va checkpoint mang tinh user-specific.

Vi vay, cach an toan nhat la luon truyen `--run`.

### Cac tham so chung

Tat ca script `predict_*` ho tro:

- `--config PATH`
- `--run PATH`
- `--model checkpoint_name.zip`
- `--episodes N`
- `--mode static|random`
- `--gui true|false`
- `--show true|false`
- `--deterministic true|false`
- `--sleep FLOAT`
- `--start-mode config|fixed|random`
- `--target X Y Z`

### Vi du cho SAC

Random evaluation:

```powershell
python .\Evaluation\predict_model_sac.py --config .\config\environment.yaml --run .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

Static evaluation:

```powershell
python .\Evaluation\predict_model_sac.py --config .\config\environment.yaml --run .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode static --episodes 3 --target 0.40 0.00 0.10 --gui false --show true
```

### Vi du cho DDPG

```powershell
python .\Evaluation\predict_model_ddpg.py --config .\config\environment.yaml --run .\Data\Training\DDPG\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

### Vi du cho PPO

```powershell
python .\Evaluation\predict_model_ppo.py --config .\config\environment.yaml --run .\Data\Training\PPO\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

### Vi du cho TD3

```powershell
python .\Evaluation\predict_model.py --config .\config\environment.yaml --run .\Data\Training\TD3\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

### Dau ra

`predict_common.py` hien ghi dau ra theo 2 kieu:

- `mode=random`: `Data/Prediction/metrics_random_N_<episodes>.csv`
- `mode=static`: `Data/Prediction/Environment_Default/<ALGO>/FRAME_ONLY/trajectory_static_ep_XXX.csv` va file `.json` di kem

`--show true` se in tong hop episode va summary ra terminal.

## Plot Va Tim Checkpoint Tot Nhat

### `plot_results.py`

Script nay doc:

- `logs/progress.csv`
- hoac `logs/monitor.csv`
- hoac evaluation CSV

Va luu plot vao:

```text
Data/Plots/<run_name>/
```

Lenh mau:

```powershell
python .\Evaluation\plot_results.py --run .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS --show false
```

Hoac:

```powershell
python .\Evaluation\plot_results.py --progress-csv .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS\logs\progress.csv --monitor .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS\logs\monitor.csv --show false
```

### `find_best_checkpoint.py`

Script nay doc `logs/progress.csv`, chon checkpoint tot nhat theo:

1. `rollout/success_rate`
2. tie-break bang `rollout/ep_rew_mean`

Lenh tim checkpoint:

```powershell
python .\Evaluation\find_best_checkpoint.py --run .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS
```

Lenh tim va evaluate:

```powershell
python .\Evaluation\find_best_checkpoint.py --run .\Data\Training\TD3\FRAME_ONLY\run_YYYYMMDD_HHMMSS --evaluate --episodes 20
```

Luu y:

- che do `--evaluate` hien goi `predict_model.py` (TD3 script)
- vi vay no hop nhat cho run TD3
- neu run DDPG/SAC/PPO, nen dung script nay de tim checkpoint, sau do goi script predict dung algorithm

## Audit Observation Va Reward

### `obs_check_cases.py`

Audit 4 truong hop obstacle:

- fixed size, fixed center
- random center
- random size
- random center + random size

Chay:

```powershell
python .\Evaluation\obs_check_cases.py
```

### `test_obs_reward_manual.py`

Script de test reward bang scene override va trajectory script tay.

```powershell
python .\Evaluation\test_obs_reward_manual.py --gui false --interactive false --scene hard --run-hard-trajs true
```

Dung khi ban muon xem reward phan ung the nao voi mot scene cu the ma khong can train.

## Script Mang Tinh Legacy / Can Xac Nhan Truoc Khi Dung

### `test_reward_strategy.py`

Script nay duoc viet cho mot reward strategy phuc tap hon reward hien tai. No van huu ich de doc y tuong test trajectory, nhung khong phai script dau tay de chay khi reward da doi.

### `analyze_reward_trajectories.py`

Script nay nghieng ve reward corridor / clearance strategy cu. Neu ban dang dung reward simple distance-based hien tai, nen doc lai source truoc khi chay va chuan bi sua mot vai field neu can.

## Ghi Chu Quan Trong

- Thu muc `Evaluation/` hien khong con file `check_real_setup_pybullet.py`.
- Script thay the thuc te de kiem tra moi truong truoc khi train la `test_environment_start_to_target.py`.
- Neu ban muon danh gia model, dung `--run` ro rang thay vi tin vao default.
- Neu config `environment.yaml` thay doi workspace/start/target/obstacle, nen chay `test_environment_start_to_target.py` it nhat vai episode truoc khi train dai.
