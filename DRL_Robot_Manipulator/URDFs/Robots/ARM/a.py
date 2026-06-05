import time
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as pb
import pybullet_data as pbd


# ============================ Tiện ích ============================

def _normalize_mesh_path(raw: str, urdf_dir: Path) -> str:
    s = raw.replace("\\", "/").strip()
    if s.startswith("package://"):
        rest = s[len("package://"):]
        s = rest.split("/", 1)[1] if "/" in rest else ""
    if s.startswith("/"):
        s = s[1:]
    if s.startswith("Mesh/"):
        s = "Mesh/" + s.split("/", 1)[1]
    s = s.replace("/visual/", "/Visual/").replace("/collision/", "/Collision/")
    s = s.replace("/VISUAL/", "/Visual/").replace("/COLLISION/", "/Collision/")
    return (urdf_dir / s).resolve().as_posix()


def resolve_urdf_meshes(urdf_path: Path) -> Path:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    changed = 0
    for mesh in root.findall(".//mesh"):
        fname = mesh.get("filename")
        if not fname:
            continue
        new_name = _normalize_mesh_path(fname, urdf_dir)
        if new_name != fname:
            mesh.set("filename", new_name)
            changed += 1
    out = Path(tempfile.mkdtemp(prefix="pb_resolved_")) / "resolved.urdf"
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    print(f"[INFO] Wrote resolved URDF -> {out} (changed {changed} mesh paths)")
    return out


def list_missing_meshes(urdf_path: Path):
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    missing = []
    for mesh in root.findall(".//mesh"):
        fname = mesh.get("filename")
        if fname and not Path(fname).exists():
            missing.append(fname)
    return missing


def add_joint_sliders(body_id: int):
    sliders = []
    for j in range(pb.getNumJoints(body_id)):
        ji = pb.getJointInfo(body_id, j)
        jtype = ji[2]
        if jtype in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
            name = ji[1].decode(errors="ignore")
            lo, hi = ji[8], ji[9]
            if lo >= hi or (abs(lo) + abs(hi) < 1e-8):
                lo, hi = -3.14159, 3.14159
            sid = pb.addUserDebugParameter(f"{j}:{name}", lo, hi, 0.0)
            sliders.append((j, sid))
    return sliders


# ============================ Link table utilities ============================

def print_link_table(body_id: int, title: str = "ROBOT") -> dict:
    print(f"\n=== {title} LINK TABLE ===")
    print("base index = -1, link name = base_link")
    name_to_index = {}
    for i in range(pb.getNumJoints(body_id)):
        ji = pb.getJointInfo(body_id, i)
        joint_name = ji[1].decode(errors="ignore")
        joint_type = ji[2]
        child_link_name = ji[12].decode(errors="ignore")
        parent_index = ji[16]
        type_name = {
            pb.JOINT_REVOLUTE: "REVOLUTE",
            pb.JOINT_PRISMATIC: "PRISMATIC",
            pb.JOINT_FIXED: "FIXED",
        }.get(joint_type, str(joint_type))
        print(
            f"idx={i:2d} | joint={joint_name:10s} | "
            f"type={type_name:9s} | child_link={child_link_name:10s} | "
            f"parent_idx={parent_index}"
        )
        name_to_index[child_link_name] = i
    return name_to_index


def find_link_index_by_name(body_id: int, link_name: str):
    for i in range(pb.getNumJoints(body_id)):
        ji = pb.getJointInfo(body_id, i)
        child_link_name = ji[12].decode(errors="ignore")
        if child_link_name == link_name:
            return i
    return None


def get_link_pose(body_id: int, link_index: int):
    ls = pb.getLinkState(body_id, link_index, computeForwardKinematics=True)
    if ls is None:
        return None
    if len(ls) >= 6 and ls[4] is not None and ls[5] is not None:
        pos = ls[4]
        orn_xyzw = ls[5]
    else:
        pos = ls[0]
        orn_xyzw = ls[1]
    orn_wxyz = [orn_xyzw[3], orn_xyzw[0], orn_xyzw[1], orn_xyzw[2]]
    rpy = pb.getEulerFromQuaternion(orn_xyzw)
    return pos, orn_xyzw, orn_wxyz, rpy


# ============================ Vẽ các trục tọa độ ============================

def _vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]

def _vec_scale(v, s):
    return [v[0] * s, v[1] * s, v[2] * s]

def draw_single_link_axis(body_id: int, link_index: int, label: str,
                           length: float = 0.12, thickness: float = 4.0,
                           prev_line_ids=None, prev_text_id=None):
    """
    Vẽ hệ trục XYZ tại một link cụ thể. Trả về (new_line_ids, text_id).
    """
    if prev_line_ids:
        for lid in prev_line_ids:
            try:
                pb.removeUserDebugItem(lid)
            except Exception:
                pass
    if prev_text_id is not None:
        try:
            pb.removeUserDebugItem(prev_text_id)
        except Exception:
            pass

    ls = pb.getLinkState(body_id, link_index, computeForwardKinematics=True)
    if ls is None:
        return [], None

    if len(ls) >= 6 and ls[4] is not None and ls[5] is not None:
        pos = ls[4]
        orn = ls[5]
    else:
        pos = ls[0]
        orn = ls[1]

    mat = pb.getMatrixFromQuaternion(orn)
    x_axis = [mat[0], mat[3], mat[6]]
    y_axis = [mat[1], mat[4], mat[7]]
    z_axis = [mat[2], mat[5], mat[8]]

    x_end = _vec_add(pos, _vec_scale(x_axis, length))
    y_end = _vec_add(pos, _vec_scale(y_axis, length))
    z_end = _vec_add(pos, _vec_scale(z_axis, length))

    new_ids = []
    try:
        lid_x = pb.addUserDebugLine(pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=thickness, lifeTime=0)
        lid_y = pb.addUserDebugLine(pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=thickness, lifeTime=0)
        lid_z = pb.addUserDebugLine(pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=thickness, lifeTime=0)
        new_ids.extend([lid_x, lid_y, lid_z])

        text_id = pb.addUserDebugText(
            label,
            [pos[0], pos[1], pos[2] + length * 0.6],
            textColorRGB=[1, 1, 1],
            textSize=1.2,
            lifeTime=0,
        )
    except Exception:
        return [], None

    return new_ids, text_id


def draw_joint_axes(body_id: int, length: float = 0.08, thickness: float = 2.0, prev_line_ids=None):
    """
    Vẽ hệ trục X (đỏ), Y (xanh lá), Z (xanh dương) cho mỗi link/khớp.
    Trả về danh sách debug line ids mới (để xoá ở vòng lặp tiếp theo).
    """
    # Xoá các line cũ (nếu có)
    if prev_line_ids:
        for lid in prev_line_ids:
            try:
                pb.removeUserDebugItem(lid)
            except Exception:
                pass

    new_ids = []
    num_joints = pb.getNumJoints(body_id)

    # Lặp qua từng link_index tương ứng với joint index trong PyBullet
    # (bỏ qua base nếu không cần vẽ; nếu muốn vẽ base hãy thêm nó)
    for link_index in range(num_joints):
        # getLinkState: indices 4 and 5 correspond to worldLinkFramePosition, worldLinkFrameOrientation
        ls = pb.getLinkState(body_id, link_index, computeForwardKinematics=True)
        # some PyBullet builds return fewer fields — fallback:
        if ls is None:
            continue
        # Prefer worldLinkFramePosition/orientation (index 4/5). If not present, dùng index 0/1.
        if len(ls) >= 6 and ls[4] is not None and ls[5] is not None:
            pos = ls[4]
            orn = ls[5]
        else:
            pos = ls[0]
            orn = ls[1]

        # Chuyển quaternion -> ma trận 3x3 (list length 9)
        mat = pb.getMatrixFromQuaternion(orn)
        # ma trận trả về: [r11,r12,r13, r21,r22,r23, r31,r32,r33]
        # vector trục x = (r11, r21, r31) -> indices 0,3,6
        x_axis = [mat[0], mat[3], mat[6]]
        y_axis = [mat[1], mat[4], mat[7]]
        z_axis = [mat[2], mat[5], mat[8]]

        # Tính điểm kết thúc cho mỗi trục
        x_end = _vec_add(pos, _vec_scale(x_axis, length))
        y_end = _vec_add(pos, _vec_scale(y_axis, length))
        z_end = _vec_add(pos, _vec_scale(z_axis, length))

        # Vẽ 3 line (X đỏ, Y xanh lá, Z xanh dương)
        try:
            lid_x = pb.addUserDebugLine(pos, x_end, lineColorRGB=[1, 0, 0], lineWidth=thickness, lifeTime=0)
            lid_y = pb.addUserDebugLine(pos, y_end, lineColorRGB=[0, 1, 0], lineWidth=thickness, lifeTime=0)
            lid_z = pb.addUserDebugLine(pos, z_end, lineColorRGB=[0, 0, 1], lineWidth=thickness, lifeTime=0)
            new_ids.extend([lid_x, lid_y, lid_z])
        except Exception:
            # Nếu add debug line thất bại (hiếm khi xảy ra), bỏ qua link này
            continue

    return new_ids


# ============================ Chương trình chính ============================

def main():
    urdf_path = Path("yaskawa_gp7.urdf").resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file URDF: {urdf_path}")

    resolved_urdf = resolve_urdf_meshes(urdf_path)
    missing = list_missing_meshes(resolved_urdf)
    if missing:
        print("[WARN] Các mesh KHÔNG tìm thấy:")
        for m in missing[:20]:
            print("   ", m)
        print("=> Kiểm tra lại tên/thư mục trong 'Mesh/Visual' & 'Mesh/Collision'.")

    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.loadURDF("plane.urdf")
    pb.resetDebugVisualizerCamera(2.2, 60, -30, [0, 0, 0.8])

    use_fixed_base = True
    robot_id = pb.loadURDF(
        str(resolved_urdf),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=use_fixed_base,
        flags=(
            pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES |
            pb.URDF_USE_MATERIAL_COLORS_FROM_MTL
        ),
    )

    ghost_id = pb.loadURDF(
        str(resolved_urdf),
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useMaximalCoordinates=False,
        useFixedBase=True,
        flags=(
            pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES |
            pb.URDF_USE_MATERIAL_COLORS_FROM_MTL
        ),
    )

    pb.setCollisionFilterGroupMask(ghost_id, -1, 0, 0)
    pb.changeVisualShape(ghost_id, -1, rgbaColor=[0.0, 0.8, 0.0, 0.35])
    pb.changeDynamics(ghost_id, -1, linearDamping=0, angularDamping=0, jointDamping=0, mass=0)

    for i in range(pb.getNumJoints(ghost_id)):
        pb.setCollisionFilterGroupMask(ghost_id, i, 0, 0)
        pb.changeVisualShape(ghost_id, i, rgbaColor=[0.0, 0.9, 0.0, 0.35])
        pb.changeDynamics(ghost_id, i, linearDamping=0, angularDamping=0, jointDamping=0, mass=0)

    print("\n" + "=" * 60)
    print("[INFO] URDF expected: ee_link is 0.090 m from link_EE along local Z.")
    print("=" * 60)

    robot_links = print_link_table(robot_id, "MAIN ROBOT")
    ghost_links = print_link_table(ghost_id, "GHOST ROBOT")

    link_EE_index = find_link_index_by_name(robot_id, "link_EE")
    ee_link_index = find_link_index_by_name(robot_id, "ee_link")
    ghost_link_EE_index = find_link_index_by_name(ghost_id, "link_EE")
    ghost_ee_link_index = find_link_index_by_name(ghost_id, "ee_link")

    print(f"\n[INFO] MAIN   link_EE index = {link_EE_index}")
    print(f"[INFO] MAIN   ee_link index = {ee_link_index}")
    print(f"[INFO] GHOST  link_EE index = {ghost_link_EE_index}")
    print(f"[INFO] GHOST  ee_link index = {ghost_ee_link_index}")

    if link_EE_index is None:
        print("[ERROR] link_EE not found. Check if URDF_MERGE_FIXED_LINKS is enabled or if the URDF link name is different.")
    if ee_link_index is None:
        print("[ERROR] ee_link not found. Check if URDF_MERGE_FIXED_LINKS is enabled or if the URDF link name is different.")

    last_print_time = 0.0
    PRINT_PERIOD_SEC = 0.5

    ee_axis_line_ids = []
    ee_axis_text_id = None

    sliders = add_joint_sliders(robot_id)

    # danh sách debug line ids hiện thời (để xoá trước khi vẽ lại)
    debug_line_ids = []

    try:
        while True:
            for j, sid in sliders:
                tgt = pb.readUserDebugParameter(sid)
                pb.setJointMotorControl2(
                    robot_id, j, pb.POSITION_CONTROL,
                    targetPosition=tgt, force=200.0,
                    positionGain=0.05, velocityGain=1.0
                )
                pb.resetJointState(ghost_id, j, tgt)

            debug_line_ids = draw_joint_axes(ghost_id, length=0.09, thickness=2.0, prev_line_ids=debug_line_ids)

            now = time.time()
            if now - last_print_time >= PRINT_PERIOD_SEC:
                if link_EE_index is not None:
                    pose = get_link_pose(robot_id, link_EE_index)
                    if pose:
                        pos, orn_xyzw, orn_wxyz, rpy = pose
                        print(
                            f"[link_EE] idx={link_EE_index} | "
                            f"pos=[{pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f}] | "
                            f"quat_xyzw=[{orn_xyzw[0]:+.6f}, {orn_xyzw[1]:+.6f}, {orn_xyzw[2]:+.6f}, {orn_xyzw[3]:+.6f}] | "
                            f"quat_wxyz=[{orn_wxyz[0]:+.6f}, {orn_wxyz[1]:+.6f}, {orn_wxyz[2]:+.6f}, {orn_wxyz[3]:+.6f}] | "
                            f"rpy=[{rpy[0]:+.6f}, {rpy[1]:+.6f}, {rpy[2]:+.6f}]"
                        )

                if ee_link_index is not None:
                    pose = get_link_pose(robot_id, ee_link_index)
                    if pose:
                        pos, orn_xyzw, orn_wxyz, rpy = pose
                        print(
                            f"[ee_link] idx={ee_link_index} | "
                            f"pos=[{pos[0]:+.6f}, {pos[1]:+.6f}, {pos[2]:+.6f}] | "
                            f"quat_xyzw=[{orn_xyzw[0]:+.6f}, {orn_xyzw[1]:+.6f}, {orn_xyzw[2]:+.6f}, {orn_xyzw[3]:+.6f}] | "
                            f"quat_wxyz=[{orn_wxyz[0]:+.6f}, {orn_wxyz[1]:+.6f}, {orn_wxyz[2]:+.6f}, {orn_wxyz[3]:+.6f}] | "
                            f"rpy=[{rpy[0]:+.6f}, {rpy[1]:+.6f}, {rpy[2]:+.6f}]"
                        )

                if link_EE_index is not None and ee_link_index is not None:
                    link_EE_pose = get_link_pose(robot_id, link_EE_index)
                    ee_link_pose = get_link_pose(robot_id, ee_link_index)
                    if link_EE_pose and ee_link_pose:
                        link_EE_pos = link_EE_pose[0]
                        ee_link_pos = ee_link_pose[0]
                        dist = np.linalg.norm(np.array(ee_link_pos) - np.array(link_EE_pos))
                        print(f"[INFO] |ee_link - link_EE| = {dist:.6f} m")

                last_print_time = now

            if ghost_ee_link_index is not None:
                ee_axis_line_ids, ee_axis_text_id = draw_single_link_axis(
                    ghost_id, ghost_ee_link_index, "ee_link / TCP",
                    length=0.14, thickness=4.0,
                    prev_line_ids=ee_axis_line_ids, prev_text_id=ee_axis_text_id,
                )
            elif ghost_link_EE_index is not None:
                ee_axis_line_ids, ee_axis_text_id = draw_single_link_axis(
                    ghost_id, ghost_link_EE_index, "link_EE",
                    length=0.10, thickness=3.0,
                    prev_line_ids=ee_axis_line_ids, prev_text_id=ee_axis_text_id,
                )

            pb.stepSimulation()
            time.sleep(1/240)
    except KeyboardInterrupt:
        pass
    finally:
        # xoá các line còn lại
        for lid in debug_line_ids:
            try:
                pb.removeUserDebugItem(lid)
            except Exception:
                pass
        for lid in ee_axis_line_ids:
            try:
                pb.removeUserDebugItem(lid)
            except Exception:
                pass
        if ee_axis_text_id is not None:
            try:
                pb.removeUserDebugItem(ee_axis_text_id)
            except Exception:
                pass
        pb.disconnect()


if __name__ == "__main__":
    main()
