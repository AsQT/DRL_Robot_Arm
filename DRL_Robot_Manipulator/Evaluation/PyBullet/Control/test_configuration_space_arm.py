# System
import argparse
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_script_dir, '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import pybullet as pb
import pybullet_data

from config_loader import PROJECT_ROOT
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
import PyBullet.Utilities


# =============================================================================
# Constants
# =============================================================================
CONST_ROBOT_NAME = 'ARM'
CONST_URDF_PATH = PROJECT_ROOT / 'URDFs' / 'Robots' / CONST_ROBOT_NAME / f'{CONST_ROBOT_NAME}.urdf'
CONST_TCP_LINK_NAME = 'tcp_link'

CONST_PYBULLET_ENV_PROPERTIES = {
    'Enable_GUI': True,
    'fps': 100,
}

# ARM target/search spaces. Adjust these if your physical setup uses a different
# table height or target region.
CONST_C_SPACES = {
    'Search': {
        'center': np.array([0.20, 0.00, 0.3], dtype=np.float64),
        'size': np.array([0.25, 0.40, 0.6], dtype=np.float64),
        'color': [1.0, 0.984, 0.0],
    },
    'Target': {
        'center': np.array([0.20, 0.00, 0.1], dtype=np.float64),
        'size': np.array([0.25, 0.4, 0.2], dtype=np.float64),
        'color': [0.0, 1.0, 0.0],
    },
}

CONST_C_TYPE = 'Target'
CONST_NUM_SAMPLES = 50
CONST_PASS_TOLERANCE = 0.01
CONST_VISIBILITY_GHOST = True
CONST_SLEEP_SEC = 0.3


# =============================================================================
# Helper Functions
# =============================================================================
def get_cuboid_vertices(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = size / 2.0
    return np.array([
        [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
        [center[0] + half[0], center[1] - half[1], center[2] - half[2]],
        [center[0] + half[0], center[1] + half[1], center[2] - half[2]],
        [center[0] - half[0], center[1] + half[1], center[2] - half[2]],
        [center[0] - half[0], center[1] - half[1], center[2] + half[2]],
        [center[0] + half[0], center[1] - half[1], center[2] + half[2]],
        [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
        [center[0] - half[0], center[1] + half[1], center[2] + half[2]],
    ], dtype=np.float64)


def add_wireframe_space(c_type: str) -> None:
    cfg = CONST_C_SPACES[c_type]
    T = HTM_Cls(None, np.float64).Translation(cfg['center'])
    PyBullet.Utilities.Add_Wireframe_Cuboid(T, cfg['size'], cfg['color'], 2.5)


def find_link_index_by_name(body_id: int, link_name: str):
    for i in range(pb.getNumJoints(body_id)):
        child_link_name = pb.getJointInfo(body_id, i)[12].decode(errors='ignore')
        if child_link_name == link_name:
            return i
    return None


def get_active_joint_indices(body_id: int):
    active = []
    for i in range(pb.getNumJoints(body_id)):
        joint_type = pb.getJointInfo(body_id, i)[2]
        if joint_type in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
            active.append(i)
    return active


def get_joint_limits(body_id: int, joint_indices: list[int]):
    lower, upper, rest = [], [], []
    for idx in joint_indices:
        info = pb.getJointInfo(body_id, idx)
        lo = float(info[8])
        hi = float(info[9])
        if lo >= hi:
            lo, hi = -np.pi, np.pi
        lower.append(lo)
        upper.append(hi)
        rest.append(0.0 if lo <= 0.0 <= hi else (lo + hi) / 2.0)
    return lower, upper, rest


def reset_robot(body_id: int, joint_indices: list[int], theta=None) -> None:
    if theta is None:
        theta = [0.0] * len(joint_indices)
    for idx, value in zip(joint_indices, theta):
        pb.resetJointState(body_id, idx, float(value))


def set_ghost_visibility(body_id: int, visible: bool, color=(0.70, 0.85, 0.60, 0.35)) -> None:
    alpha = color[3] if visible else 0.0
    rgba = [color[0], color[1], color[2], alpha]
    for i in range(-1, pb.getNumJoints(body_id)):
        pb.changeVisualShape(body_id, i, rgbaColor=rgba)


def sample_target(c_type: str) -> np.ndarray:
    vertices = get_cuboid_vertices(CONST_C_SPACES[c_type]['center'], CONST_C_SPACES[c_type]['size'])
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return np.random.uniform(lower, upper)


def solve_ik(robot_id: int, ghost_id: int, tcp_link_index: int, active_joint_indices: list[int],
             target_pos: np.ndarray, pass_tolerance: float, enable_ghost: bool):
    lower, upper, rest = get_joint_limits(robot_id, active_joint_indices)
    ranges = [hi - lo for lo, hi in zip(lower, upper)]
    damping = [0.05] * len(active_joint_indices)

    theta_raw = pb.calculateInverseKinematics(
        robot_id,
        tcp_link_index,
        target_pos.tolist(),
        lowerLimits=lower,
        upperLimits=upper,
        jointRanges=ranges,
        restPoses=rest,
        jointDamping=damping,
        maxNumIterations=500,
        residualThreshold=1e-6,
    )
    theta = np.array(theta_raw[:len(active_joint_indices)], dtype=np.float64)

    if len(theta) != len(active_joint_indices) or not np.all(np.isfinite(theta)):
        set_ghost_visibility(ghost_id, enable_ghost, (0.85, 0.60, 0.60, 0.35))
        return False, theta, np.inf

    for value, lo, hi in zip(theta, lower, upper):
        if value < lo or value > hi:
            set_ghost_visibility(ghost_id, enable_ghost, (0.85, 0.60, 0.60, 0.35))
            return False, theta, np.inf

    reset_robot(ghost_id, active_joint_indices, theta)
    ghost_link_state = pb.getLinkState(ghost_id, tcp_link_index, computeForwardKinematics=True)
    ghost_tcp_pos = np.array(ghost_link_state[4], dtype=np.float64)
    position_error = float(np.linalg.norm(target_pos - ghost_tcp_pos))

    if position_error <= pass_tolerance:
        set_ghost_visibility(ghost_id, enable_ghost, (0.70, 0.85, 0.60, 0.35))
        return True, theta, position_error

    set_ghost_visibility(ghost_id, enable_ghost, (0.85, 0.60, 0.60, 0.35))
    return False, theta, position_error


def print_error_stats(title: str, errors: list, unit: str = 'm') -> None:
    print(f'  {title}:')
    if errors:
        arr = np.array(errors, dtype=np.float64)
        print(f'    count = {arr.size}')
        print(f'    min   = {float(np.min(arr)):.6f} {unit}')
        print(f'    max   = {float(np.max(arr)):.6f} {unit}')
        print(f'    mean  = {float(np.mean(arr)):.6f} {unit}')
        print(f'    std   = {float(np.std(arr)):.6f} {unit}')
    else:
        print('    no samples')


def parse_args():
    parser = argparse.ArgumentParser(description='Random configuration-space IK test for ARM.')
    parser.add_argument('--samples', type=int, default=CONST_NUM_SAMPLES)
    parser.add_argument('--c-type', choices=['Search', 'Target'], default=CONST_C_TYPE)
    parser.add_argument('--headless', action='store_true', help='Run with PyBullet DIRECT instead of GUI.')
    parser.add_argument('--no-ghost', action='store_true', help='Hide the ghost robot.')
    parser.add_argument('--sleep', type=float, default=CONST_SLEEP_SEC)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    enable_gui = CONST_PYBULLET_ENV_PROPERTIES['Enable_GUI'] and not args.headless
    if enable_gui:
        pb.connect(pb.GUI, options='--background_color_red=0.0 --background_color_green=0.0 --background_color_blue=0.0')
    else:
        pb.connect(pb.DIRECT)

    pb.resetSimulation()
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setTimeStep(1.0 / CONST_PYBULLET_ENV_PROPERTIES['fps'])
    pb.setRealTimeSimulation(0)
    pb.setGravity(0.0, 0.0, -9.81)
    pb.loadURDF('plane.urdf')

    robot_id = pb.loadURDF(str(CONST_URDF_PATH), useFixedBase=True, flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    ghost_id = pb.loadURDF(str(CONST_URDF_PATH), useFixedBase=True, flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    active_joint_indices = get_active_joint_indices(robot_id)
    tcp_link_index = find_link_index_by_name(robot_id, CONST_TCP_LINK_NAME)
    ghost_tcp_link_index = find_link_index_by_name(ghost_id, CONST_TCP_LINK_NAME)

    if tcp_link_index is None or ghost_tcp_link_index is None:
        raise RuntimeError(f'Cannot find TCP link "{CONST_TCP_LINK_NAME}" in {CONST_URDF_PATH}')

    for i in range(-1, pb.getNumJoints(robot_id)):
        pb.changeVisualShape(robot_id, i, rgbaColor=[0.72, 0.74, 0.76, 1.0])
    for i in range(-1, pb.getNumJoints(ghost_id)):
        pb.setCollisionFilterGroupMask(ghost_id, i, 0, 0)
    set_ghost_visibility(ghost_id, not args.no_ghost)

    reset_robot(robot_id, active_joint_indices)
    reset_robot(ghost_id, active_joint_indices)
    add_wireframe_space('Search')
    add_wireframe_space('Target')

    target_vertices = get_cuboid_vertices(CONST_C_SPACES[args.c_type]['center'], CONST_C_SPACES[args.c_type]['size'])
    print(f'Robot: {CONST_ROBOT_NAME}')
    print(f'URDF : {CONST_URDF_PATH}')
    print(f'TCP link: {CONST_TCP_LINK_NAME} -> index {tcp_link_index}')
    print(f'Active joints: {active_joint_indices}')
    print(f'{args.c_type} C-space bounds (x, y, z):')
    print(f'  min = {target_vertices.min(axis=0)}')
    print(f'  max = {target_vertices.max(axis=0)}')
    print(f'Random samples: {args.samples}')
    print(f'IK: position-only, tol={CONST_PASS_TOLERANCE:.3f} m, iters=500')
    print()

    count_ik_failed = 0
    count_passed = 0
    count_pose_mismatch = 0
    pos_errors_all = []
    pos_errors_passed = []
    pos_errors_failed = []
    failed_positions = []

    for sample in range(args.samples):
        target_pos = sample_target(args.c_type)
        successful, theta, ghost_error = solve_ik(
            robot_id,
            ghost_id,
            tcp_link_index,
            active_joint_indices,
            target_pos,
            CONST_PASS_TOLERANCE,
            not args.no_ghost,
        )

        if not successful:
            print(f'[{sample:4d}/{args.samples}] [IK FAILED]  '
                  f'ghost_err={ghost_error:.6f} m  target={target_pos}  theta={theta}')
            count_ik_failed += 1
            failed_positions.append(target_pos.copy())
            time.sleep(args.sleep)
            reset_robot(robot_id, active_joint_indices)
            reset_robot(ghost_id, active_joint_indices)
            continue

        reset_robot(robot_id, active_joint_indices, theta)
        pb.stepSimulation()
        link_state = pb.getLinkState(robot_id, tcp_link_index, computeForwardKinematics=True)
        tcp_pos = np.array(link_state[4], dtype=np.float64)
        pos_error = float(np.linalg.norm(target_pos - tcp_pos))

        if pos_error > CONST_PASS_TOLERANCE:
            result_tag = 'POSE_MISMATCH'
            count_pose_mismatch += 1
            pos_errors_failed.append(pos_error)
        else:
            result_tag = 'PASS'
            count_passed += 1
            pos_errors_passed.append(pos_error)

        pos_errors_all.append(pos_error)
        print(f'[{sample:4d}/{args.samples}] [{result_tag}]  '
              f'pos_err={pos_error:.6f} m  ghost_err={ghost_error:.6f} m  target={target_pos}')

        time.sleep(args.sleep)
        reset_robot(robot_id, active_joint_indices)
        reset_robot(ghost_id, active_joint_indices)

    total = args.samples
    print()
    print('=' * 60)
    print('=== ARM RANDOM TEST SUMMARY ===')
    print('=' * 60)
    print(f'  Total samples            : {total}')
    print(f'  IK FAILED                : {count_ik_failed}')
    print(f'  PASS                     : {count_passed}')
    print(f'  POSE_MISMATCH            : {count_pose_mismatch}')
    print(f'  Counter sum              : {count_ik_failed + count_passed + count_pose_mismatch}  (should == {total})')
    print()
    print('--- Position Error Statistics ---')
    print_error_stats('ALL EXECUTED position errors', pos_errors_all, 'm')
    print_error_stats('PASS position errors', pos_errors_passed, 'm')
    print_error_stats('FAILED executed position errors', pos_errors_failed, 'm')

    if failed_positions:
        arr = np.array(failed_positions)
        print()
        print('  IK-FAILED target positions (x, y, z):')
        print(f'    count = {arr.shape[0]}')
        print(f'    min   = {arr.min(axis=0)}')
        print(f'    max   = {arr.max(axis=0)}')
        print(f'    mean  = {arr.mean(axis=0)}')
    else:
        print()
        print('  IK-FAILED target positions: none')

    if enable_gui:
        print()
        print('Test complete. Closing GUI.')

    if pb.isConnected():
        pb.disconnect()


if __name__ == '__main__':
    sys.exit(main())
