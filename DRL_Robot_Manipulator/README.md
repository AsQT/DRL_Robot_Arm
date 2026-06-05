# DRL Robot Manipulator

Project này là framework huấn luyện và đánh giá Deep Reinforcement Learning cho robot manipulator trong PyBullet. Môi trường sử dụng Gymnasium làm API, Stable-Baselines3 cho các thuật toán DRL, và thư viện `RoLE` để hỗ trợ kinematics, transformation, trajectory, collider.

Hiện project có hai bộ robot URDF chính:

- `YASKAWA_GP7`: robot Yaskawa GP7 dùng trong các script training/evaluation hiện có.
- `ARM`: robot arm được chuyển từ package `robot_description`, gồm 6 revolute joints, 2 prismatic gripper joints, `tcp_link` và camera `astra_link`.

## Mục tiêu

Agent học cách điều khiển TCP đi tới target trong không gian làm việc. Project hỗ trợ hai chế độ môi trường:

| Mode | Mô tả |
|---|---|
| `Default` | Reaching trong không gian tự do, không có vật cản. |
| `Collision-Free` | Reaching có vật cản, policy cần tránh collision. |

## Cấu trúc thư mục

```text
DRL_Robot_Manipulator/
├── src/
│   ├── config_loader.py
│   ├── Industrial_Robotics_Gym/
│   │   └── Environment/
│   │       ├── GP7ReachPyBulletEnv.py
│   │       └── Core.py
│   ├── PyBullet/
│   │   ├── Core.py
│   │   ├── Utilities.py
│   │   └── Configuration/
│   └── RoLE/
│       ├── Parameters/
│       ├── Transformation/
│       ├── Kinematics/
│       ├── Collider/
│       ├── Primitives/
│       ├── Trajectory/
│       └── Interpolation/
├── Training/
│   ├── train_ddpg_gp7.py
│   ├── train_ddpg.py
│   ├── train_sac.py
│   └── train_td3.py
├── Evaluation/
│   ├── Gym/
│   └── PyBullet/
├── URDFs/
│   ├── Robots/
│   │   ├── ARM/
│   │   └── YASKAWA_GP7/
│   ├── Primitives/
│   └── Viewpoint/
├── Textures/
├── config/
├── Data/
└── requirements.txt
```

## Thành phần chính

| Thành phần | Vai trò |
|---|---|
| `src/Industrial_Robotics_Gym` | Định nghĩa Gymnasium environment cho bài toán reaching. |
| `src/PyBullet` | Interface load robot, mô phỏng và hiển thị trong PyBullet. |
| `src/RoLE` | Robotics helper library: FK/IK, transformation, trajectory, collider, primitives. |
| `Training` | Script train DDPG, SAC, TD3. |
| `Evaluation` | Script predict, plot kết quả training, kiểm tra IK/config space. |
| `URDFs` | Robot URDF, mesh, primitive object, viewpoint frame. |
| `Data` | Output training/evaluation như model, log, prediction. |

## Robot URDF

### `URDFs/Robots/YASKAWA_GP7`

Chứa URDF và mesh cho robot Yaskawa GP7. Đây là robot đang được các script hiện có sử dụng mặc định.

File chính:

- `YASKAWA_GP7.urdf`
- `Mesh/Visual/*.stl`
- `Mesh/Collision/*.STL`

### `URDFs/Robots/ARM`

Chứa URDF và mesh cho robot `ARM`, được chuyển từ package `robot_description`.

File chính:

- `ARM.urdf`
- `Mesh/Visual/base_link.stl`, `link_1.stl` đến `link_6.stl`
- `Mesh/Visual/gripper_l.stl`, `gripper_r.stl`, `astra_cam.stl`
- `Mesh/Collision/...` tương ứng cho collision geometry

`ARM.urdf` đã được chuyển sang URDF thuần để PyBullet có thể load trực tiếp, không cần xacro.

## Cài đặt

Tạo môi trường:

```powershell
cd DRL_Robot_Manipulator
python -m venv .venv
```

Kích hoạt môi trường:

| Terminal | Lệnh |
|---|---|
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| CMD | `.venv\Scripts\activate.bat` |
| Git Bash | `source .venv/Scripts/activate` |
| Linux/macOS | `source .venv/bin/activate` |

Cài dependency:

```powershell
pip install -r requirements.txt
```

Nếu cần GPU, cài PyTorch CUDA trước khi cài các package còn lại:

```powershell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Kiểm tra dependency:

```powershell
python -c "import gymnasium; import stable_baselines3; import pybullet; print('OK')"
```

## Chạy training

Từ thư mục `DRL_Robot_Manipulator`:

```powershell
python Training\train_ddpg_gp7.py
python Training\train_ddpg.py
python Training\train_sac.py
python Training\train_td3.py
```

Output training thường nằm trong:

```text
Data/Training/Environment_{MODE}/{ALGORITHM}/{ROBOT_NAME}/run_{TIMESTAMP}/
├── config.json
├── model/
│   ├── final_model.zip
│   └── best_model.zip
└── logs/
    ├── progress.csv
    ├── monitor.csv
    └── time.txt
```

Các output này thường không nên commit lên Git, trừ khi cần lưu kết quả báo cáo hoặc demo.

## Environment Design

### Action Space

`Box(-1.0, 1.0, shape=(3,))`

Action là delta TCP trong không gian Cartesian:

```text
[dx, dy, dz]
```

Environment scale action bằng `action_step` để tạo displacement thực tế mỗi step.

### Observation Space

Observation là vector 15 chiều:

| Index | Field | Mô tả |
|---|---|---|
| `0-2` | `tcp_x/y/z` | Vị trí TCP hiện tại. |
| `3-5` | `target_x/y/z` | Vị trí target. |
| `6-8` | `err_x/y/z` | Sai số `target - tcp`. |
| `9-11` | `rel_obs_x/y/z` | Vị trí vật cản tương đối với TCP. |
| `12-14` | `obs_size_x/y/z` | Kích thước/half extent của vật cản. |

### Termination

| Điều kiện | Kết quả |
|---|---|
| TCP cách target dưới `0.01 m` | `terminated = True` |
| Vượt workspace, IK fail, joint limit, collision, max steps | `truncated = True` |

## Evaluation

Một số nhóm script đánh giá:

| Thư mục/script | Mục đích |
|---|---|
| `Evaluation/Gym/Model/Prediction/Static` | Predict tới target cố định. |
| `Evaluation/Gym/Model/Prediction/Random` | Predict nhiều target random. |
| `Evaluation/Gym/Model/Training` | Plot/so sánh kết quả training. |
| `Evaluation/PyBullet/Control` | Kiểm tra IK và configuration space. |

Kiểm tra configuration space cho robot `ARM`:

```powershell
python Evaluation\PyBullet\Control\test_configuration_space_arm.py
```

Chạy nhanh không mở GUI:

```powershell
python Evaluation\PyBullet\Control\test_configuration_space_arm.py --headless --samples 10 --sleep 0
```

Kiểm tra toàn bộ vertices của target/search space cho `ARM`:

```powershell
python Evaluation\PyBullet\Control\test_configuration_space_vertices_arm.py
```

Chạy nhanh không mở GUI:

```powershell
python Evaluation\PyBullet\Control\test_configuration_space_vertices_arm.py --headless --sleep 0
```

## Ghi chú phát triển

- URDF và mesh là asset cần thiết để mô phỏng, nên commit lên Git.
- `venv`, `__pycache__`, model `.zip`, replay buffer `.pkl`, TensorBoard events và log training không nên commit.
- Nếu PyBullet báo lỗi không tìm thấy mesh, kiểm tra lại path trong URDF và vị trí file trong `URDFs/Robots/<ROBOT_NAME>/Mesh`.
- Các script hiện tại vẫn ưu tiên `YASKAWA_GP7`; nếu muốn dùng `ARM` trong training, cần cập nhật phần chọn robot/URDF trong code load robot.
