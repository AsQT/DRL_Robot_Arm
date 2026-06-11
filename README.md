# DRL Robot Arm Projects

Repository này gồm hai project Deep Reinforcement Learning liên quan đến lập kế hoạch đường đi và điều khiển robot arm. Hai project có thể dùng độc lập, nhưng cùng phục vụ mục tiêu chung: huấn luyện agent DRL để sinh chuyển động/đường đi an toàn cho robot trong không gian 3D.

## Cấu trúc chính

```text
DRL/
├── DRL_Pathplanning_trainning/   # Project 1: huấn luyện path planning dạng điểm trong không gian Cartesian
├── DRL_Robot_Manipulator/        # Project 2: huấn luyện robot manipulator trong PyBullet
├── requirements.txt              # Dependency chung ở root, nếu cần
└── .gitignore                    # Bỏ qua venv, cache Python, model/log training
```

## 1. DRL_Pathplanning_trainning

`DRL_Pathplanning_trainning` là project huấn luyện DRL cho bài toán Cartesian path planning. Agent được mô hình hóa như một điểm ảo trong không gian 3D, nhận action dạng delta `[x, y, z]` và học cách di chuyển từ điểm bắt đầu đến target.

Project này phù hợp cho tầng lập kế hoạch đường đi trừu tượng:

- Không dùng mô hình robot thật trong lúc train.
- Không chạy IK/FK, joint control hoặc MoveIt.
- Có thể dùng PyBullet để visualize workspace, target, obstacle và path.
- Kết quả chính là danh sách waypoint Cartesian có thể đưa sang tầng robot execution.

Các thành phần chính:

- `src/drl_pathplanning/`: môi trường Gymnasium, reward, collision geometry, training utilities.
- `Training/`: script train DDPG, SAC, TD3.
- `Evaluation/`: script test, predict, visualize, plot kết quả.
- `config/environment.yaml`: cấu hình workspace, target, obstacle, reward, visualization.
- `requirements.txt`: dependency riêng cho project path planning.

Chạy train ví dụ:

```powershell
cd DRL_Pathplanning_trainning
python -m venv venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Training\train_ddpg.py --config config\environment.yaml
```

## 2. DRL_Robot_Manipulator

`DRL_Robot_Manipulator` là project huấn luyện DRL trực tiếp trên robot manipulator trong PyBullet. Agent học cách điều khiển TCP của robot đi tới target trong không gian làm việc, có hoặc không có vật cản.

Project này gần với mô phỏng robot thật hơn:

- Dùng URDF và mesh của robot Yaskawa GP7 và robot `ARM`.
- Dùng PyBullet cho simulation/rendering.
- Dùng Gymnasium để đóng gói environment.
- Dùng Stable-Baselines3 để train DDPG, SAC, TD3.
- Có thư viện `RoLE` hỗ trợ kinematics, transformation, trajectory, collider.

Các thành phần chính:

- `src/Industrial_Robotics_Gym/`: Gymnasium environment cho robot manipulator.
- `src/PyBullet/`: interface mô phỏng robot bằng PyBullet.
- `src/RoLE/`: robotics library gồm kinematics, collider, interpolation, trajectory.
- `Training/`: script train các thuật toán DRL.
- `Evaluation/`: script predict, plot training result, kiểm tra IK/config space.
- `URDFs/`: URDF và mesh của `YASKAWA_GP7`, `ARM`, primitive objects, viewpoint.
- `Data/`: output train/predict, model, log, replay buffer.

Chạy train ví dụ:

```powershell
cd DRL_Robot_Manipulator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Training\train_ddpg_gp7.py
```

## So sánh nhanh hai project

| Nội dung | DRL_Pathplanning_trainning | DRL_Robot_Manipulator |
|---|---|---|
| Mục tiêu | Sinh waypoint Cartesian | Điều khiển TCP robot manipulator |
| Đối tượng train | Point-agent 3D | Robot manipulator trong PyBullet |
| Robot URDF | Không dùng | Có dùng |
| IK/FK | Không dùng trong training | Có liên quan qua robot simulation/RoLE |
| PyBullet | Visualization tùy chọn | Simulation chính |
| Output | Waypoint/path | Model điều khiển robot, log, prediction |
| Thuật toán | DDPG, SAC, TD3 | DDPG, SAC, TD3 |

## Cài đặt chung

Nên tạo virtual environment riêng và không push thư mục môi trường ảo lên Git.

Tạo môi trường:

```powershell
python -m venv .venv
```

Kích hoạt môi trường:

| Terminal | Lệnh |
|---|---|
| PowerShell | `.\venv\Scripts\Activate.ps1` |
| CMD | `venv\Scripts\activate.bat` |
| Git Bash | `source venv/Scripts/activate` |
| Linux/macOS | `source venv/bin/activate` |

Cài dependency:

```powershell
pip install -r requirements.txt
```

Nếu từng project có `requirements.txt` riêng, nên cài dependency trong đúng thư mục project đó để tránh thiếu package.

## Những file không nên push

Repo đã cấu hình `.gitignore` để bỏ qua các file sinh tự động hoặc file output lớn:

- `venv/`, `.venv/`, `env/`, `ENV/`
- `__pycache__/`, `*.pyc`
- `*.zip` model train như `best_model.zip`, `final_model.zip`
- `*.pkl` replay buffer hoặc normalize stats
- TensorBoard events và log training như `monitor.csv`, `progress.csv`, `time.txt`

Các file source code, config, README, URDF, mesh và texture cần để chạy mô phỏng thì nên được commit.

## Ghi chú

- Xem README riêng trong từng project để biết chi tiết hơn về architecture, config và cách chạy.
- Nếu train bằng GPU, cần cài đúng bản PyTorch CUDA phù hợp với máy.
- Các output trong `Data/Training` và `Data/Prediction` thường là kết quả thực nghiệm; chỉ commit khi cần lưu kết quả báo cáo hoặc demo.


Từ repo root C:\Users\MinhQuang\DRL:
& .\venv\Scripts\python.exe .\DRL_Pathplanning_trainning\Evaluation\test_environment_start_to_target.py --config .\DRL_Pathplanning_trainning\config\environment.yaml --episodes 3 --steps 50 --gui true --show false
Hoặc vào đúng thư mục project trước:
cd .\DRL_Pathplanning_trainning
& ..\venv\Scripts\python.exe .\Evaluation\test_environment_start_to_target.py --config .\config\environment.yaml --episodes 3 --steps 50 --gui false --show false