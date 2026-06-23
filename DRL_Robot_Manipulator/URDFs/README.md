# URDFs

Thu muc `URDFs` chua cac file URDF va mesh dung de load robot, vat the, mat phang va frame quan sat trong PyBullet cho project `DRL_Robot_Manipulator`.

## Cau truc

```text
URDFs/
+-- Robots/
|   +-- ARM/
|   |   +-- ARM.urdf
|   |   +-- a.py
|   |   +-- Mesh/
|   |       +-- Visual/
|   |       +-- Collision/
|   +-- YASKAWA_GP7/
|       +-- YASKAWA_GP7.urdf
|       +-- a.py
|       +-- Mesh/
|           +-- Visual/
|           +-- Collision/
+-- Primitives/
|   +-- Cube/
|   +-- Cube100/
|   +-- Plane/
|   +-- Sphere/
|   +-- Table/
+-- Viewpoint/
    +-- Viewpoint.urdf
    +-- Mesh/Visual/
```

## Robots

### `Robots/ARM`

Robot `ARM` duoc chuyen tu package `robot_description` sang URDF thuan de PyBullet load truc tiep.

Thanh phan chinh:

| File/thu muc | Mo ta |
|---|---|
| `ARM.urdf` | URDF robot ten `ARM`. |
| `Mesh/Visual/` | Mesh hien thi: `base_link`, `link_1` den `link_6`, `gripper_l`, `gripper_r`, `astra_cam`. |
| `Mesh/Collision/` | Mesh collision tuong ung. |
| `a.py` | Script ho tro/debug load robot khi can. |

Thong tin robot:

- TCP link: `tcp_link`
- Active joints: `joint_1` den `joint_6`, `joint_gl`, `joint_gr`
- `joint_1..joint_6`: revolute
- `joint_gl`, `joint_gr`: prismatic gripper
- Camera link: `astra_link`

Config cho robot `ARM` nam trong:

```text
src/RoLE/Parameters/Robot.py
src/PyBullet/Configuration/Environment.py
src/PyBullet/Utilities.py
```

Kiem tra vertices target space:

```powershell
cd C:\Users\MinhQuang\DRL
.\.venv\Scripts\Activate.ps1
python .\DRL_Robot_Manipulator\Evaluation\PyBullet\Control\test_configuration_space_vertices_arm.py --headless --sleep 0
```

Ket qua gan day:

```text
Total vertices : 8
PASS           : 8
IK FAILED      : 0
max error      : 0.002669 m
```

### `Robots/YASKAWA_GP7`

Robot `YASKAWA_GP7` la robot mac dinh cua cac script training/evaluation cu.

Thanh phan chinh:

| File/thu muc | Mo ta |
|---|---|
| `YASKAWA_GP7.urdf` | URDF robot Yaskawa GP7. |
| `Mesh/Visual/` | Mesh STL dung de hien thi robot. |
| `Mesh/Collision/` | Mesh STL dung cho collision geometry. |
| `a.py` | Script ho tro/debug load robot khi can. |

Robot nay dung TCP/link theo config hien co cua GP7.

## Primitives

Thu muc `Primitives` chua cac vat the co ban dung trong simulation.

| Thu muc | File URDF | Vai tro |
|---|---|---|
| `Cube/` | `Cube.urdf` | Khoi lap phuong tong quat. |
| `Cube100/` | `Cube100.urdf` | Khoi lap phuong 100 mm, co the dung lam obstacle. |
| `Plane/` | `Plane.urdf` | Mat phang san/nen. |
| `Sphere/` | `Sphere.urdf` | Hinh cau marker cho target/TCP/diem tham chieu. |
| `Table/` | `Table.URDF` | Ban/gia do trong scene PyBullet. |

## Viewpoint

`Viewpoint` dung de hien thi frame/truc toa do trong PyBullet.

| File | Mo ta |
|---|---|
| `Viewpoint.urdf` | URDF frame quan sat. |
| `Viewpoint.stl` | Mesh trung tam. |
| `X_Axis.stl` | Truc X. |
| `Y_Axis.stl` | Truc Y. |
| `Z_Axis.stl` | Truc Z. |

## Luu y khi chinh sua

- Khong doi ten mesh neu chua sua path trong URDF.
- Neu them robot moi, nen tao cau truc `Robots/<ROBOT_NAME>/<ROBOT_NAME>.urdf`.
- Mesh visual va collision nen tach vao `Mesh/Visual` va `Mesh/Collision`.
- URDF va mesh la asset can thiet de chay simulation, nen commit len Git.
- Khong commit output training nhu model `.zip`, replay buffer `.pkl`, log TensorBoard, `progress.csv`, `monitor.csv`.

## Kiem tra nhanh URDF ARM

```powershell
cd C:\Users\MinhQuang\DRL
.\.venv\Scripts\Activate.ps1
python -c "import pybullet as p; p.connect(p.DIRECT); r=p.loadURDF(r'DRL_Robot_Manipulator/URDFs/Robots/ARM/ARM.urdf', useFixedBase=True); print(p.getNumJoints(r)); p.disconnect()"
```

Ket qua mong doi:

```text
11
```
