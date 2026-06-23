# DRL Cartesian Path Planning Training

Project nay train agent DRL cho bai toan path planning trong khong gian Cartesian 3D. Agent la mot diem ao, nhan action dang delta `[x, y, z]`, di tu start den target va co the dung PyBullet chi de hien thi workspace, target, obstacle va path.

Tat ca lenh trong README nay danh cho **VS Code Terminal tren Windows, profile PowerShell**. Chi dung PowerShell trong VS Code cho cac lenh setup, activate, train va evaluate.

## Cau truc hien tai

```text
DRL_Pathplanning_trainning/
+-- README.md
+-- requirements.txt
+-- config/
|   +-- environment.yaml
|   +-- environment_yaskawa.yaml
+-- src/drl_pathplanning/
|   +-- geometry/
|   +-- gymnasium/
|   +-- pybullet/
|   +-- training/
+-- Training/
|   +-- train_ddpg.py
|   +-- train_sac.py
|   +-- train_td3.py
|   +-- train_ppo.py
|   +-- *_continue_checkpoint.py
+-- Evaluation/
|   +-- test_environment_start_to_target.py
|   +-- predict_model.py
|   +-- predict_model_ddpg.py
|   +-- predict_model_sac.py
|   +-- predict_model_ppo.py
|   +-- plot_results.py
|   +-- find_best_checkpoint.py
+-- Data/
```

## Moi Truong Ao

Dung chung virtual environment o repo root:

```text
C:\Users\MinhQuang\DRL\.venv
```

Mo VS Code tai repo root `C:\Users\MinhQuang\DRL`, sau do mo terminal tich hop:

```powershell
cd C:\Users\MinhQuang\DRL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\DRL_Pathplanning_trainning\requirements.txt
```

Neu terminal dang o thu muc project:

```powershell
cd C:\Users\MinhQuang\DRL\DRL_Pathplanning_trainning
..\.venv\Scripts\Activate.ps1
```

Sau khi activate dung, prompt PowerShell se co `(.venv)` o dau dong. Kiem tra interpreter:

```powershell
python -c "import sys; print(sys.executable)"
```

Ket qua nen tro ve:

```text
C:\Users\MinhQuang\DRL\.venv\Scripts\python.exe
```

## Config Chinh

File config dang dung:

```text
config\environment.yaml
```

File nay la nguon chinh cho workspace, start, target, obstacle, reward, visualization va training. File `config\environment_yaskawa.yaml` la bien the khac; cac file trong `config\deprecated\` chi nen xem lai khi can doi chieu cau hinh cu.

## Kiem Tra Moi Truong Truoc Khi Train

Chay tu VS Code Terminal:

```powershell
cd C:\Users\MinhQuang\DRL
.\.venv\Scripts\Activate.ps1
cd .\DRL_Pathplanning_trainning
python .\Evaluation\test_environment_start_to_target.py --config .\config\environment.yaml --episodes 3 --steps 50 --gui false --show false
```

Neu muon mo PyBullet GUI de xem truc quan:

```powershell
python .\Evaluation\test_environment_start_to_target.py --config .\config\environment.yaml --episodes 3 --steps 50 --gui true --show false
```

Script nay dung de kiem tra:

- start/target co nam dung vung khong
- obstacle co dat dung vi tri khong
- duong di thang co bi collision/workspace violation khong
- visualization co khop voi config khong

## Train Model

Chay trong thu muc `DRL_Pathplanning_trainning` sau khi da activate `.venv`.

```powershell
python .\Training\train_ddpg.py --config .\config\environment.yaml
python .\Training\train_sac.py --config .\config\environment.yaml
python .\Training\train_td3.py --config .\config\environment.yaml
python .\Training\train_ppo.py --config .\config\environment.yaml
```

Train nhanh de test pipeline:

```powershell
python .\Training\train_sac.py --config .\config\environment.yaml --timesteps 5000 --gui false --show false
```

Debug co GUI nen chay it episode/steps va de `training.n_envs: 1` trong config:

```powershell
python .\Training\train_td3.py --config .\config\environment.yaml --timesteps 5000 --gui true --show true --render-sleep 0.02 --render-first-episodes 3
```

## Continue Tu Checkpoint

Project co cac script continue rieng:

```powershell
python .\Training\train_ddpg_continue_checkpoint.py
python .\Training\train_sac_continue_checkpoint.py
python .\Training\train_td3_continue_checkpoint.py
python .\Training\train_ppo_continue_checkpoint.py
```

Truoc khi dung, mo script tuong ung va kiem tra lai duong dan checkpoint/model trong file.

## Predict Va Evaluate

Luon truyen `--run` ro rang de tranh dung nham run cu.

```powershell
python .\Evaluation\predict_model_sac.py --config .\config\environment.yaml --run .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

```powershell
python .\Evaluation\predict_model_ddpg.py --config .\config\environment.yaml --run .\Data\Training\DDPG\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

```powershell
python .\Evaluation\predict_model_ppo.py --config .\config\environment.yaml --run .\Data\Training\PPO\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

```powershell
python .\Evaluation\predict_model.py --config .\config\environment.yaml --run .\Data\Training\TD3\FRAME_ONLY\run_YYYYMMDD_HHMMSS --mode random --episodes 10 --gui false --show true
```

Tim checkpoint tot nhat:

```powershell
python .\Evaluation\find_best_checkpoint.py --run .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS
```

Ve plot tu log:

```powershell
python .\Evaluation\plot_results.py --run .\Data\Training\SAC\FRAME_ONLY\run_YYYYMMDD_HHMMSS --show false
```

## Output

Output training nam trong `Data\Training\...`, thuong gom:

```text
model\final_model.zip
model\best_model.zip
logs\progress.csv
logs\monitor.csv
logs\time.txt
tensorboard\
```

Nhung file output lon nhu model `.zip`, replay buffer `.pkl`, TensorBoard events, `progress.csv`, `monitor.csv` va log training khong nen commit tru khi can luu ket qua bao cao/demo.

## Ghi Chu Windows/PowerShell

- Mo repo bang VS Code va chay lenh trong terminal tich hop PowerShell.
- Activate moi truong bang `.\.venv\Scripts\Activate.ps1` tu repo root, hoac `..\.venv\Scripts\Activate.ps1` tu thu muc project.
- Sau khi activate, dung `python ...` truc tiep.
- Neu PowerShell bao loi execution policy khi activate, chay:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sau do dong terminal, mo lai VS Code Terminal va activate lai `.venv`.
