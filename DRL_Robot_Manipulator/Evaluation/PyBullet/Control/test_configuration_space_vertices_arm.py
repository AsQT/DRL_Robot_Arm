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

import RoLE.Parameters.Robot as Parameters
import PyBullet.Core
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
from config_loader import PROJECT_ROOT


# =============================================================================
# Constants
# =============================================================================
CONST_ROBOT_TYPE = Parameters.ARM_Str
CONST_PROJECT_FOLDER = str(PROJECT_ROOT)
CONST_C_TYPE = 'Target'
CONST_POS_TOLERANCE = 0.01
CONST_VISIBILITY_GHOST = True

CONST_PYBULLET_ENV_PROPERTIES = {
    'Enable_GUI': True,
    'fps': 100,
    'External_Base': None,
    'Env_ID': 0,
    'Camera': {
        'Yaw': 70.0,
        'Pitch': -32.0,
        'Distance': 1.3,
        'Position': [0.05, -0.10, 0.25],
    },
}

CONST_IK_PROPERTIES = {
    'delta_time': 0.01,
    'num_of_iteration': 500,
    'tolerance': 1e-6,
    'use_orientation': False,
    'ik_position_tolerance': CONST_POS_TOLERANCE,
}


# =============================================================================
# Helper Functions
# =============================================================================
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


def print_link_table(robot_id: int) -> None:
    print('=== PYBULLET LINK/JOINT TABLE ===')
    print(f'  {"idx":>3}  {"joint_name":<28}  {"link_name":<24}  {"joint_type"}')
    print('  ' + '-' * 80)
    print(f'  {"-1":>3}  {"BASE":<28}  {"base_link":<24}  {"FIXED"}')
    type_map = {
        pb.JOINT_REVOLUTE: 'REVOLUTE',
        pb.JOINT_PRISMATIC: 'PRISMATIC',
        pb.JOINT_SPHERICAL: 'SPHERICAL',
        pb.JOINT_PLANAR: 'PLANAR',
        pb.JOINT_FIXED: 'FIXED',
    }
    for idx in range(pb.getNumJoints(robot_id)):
        info = pb.getJointInfo(robot_id, idx)
        joint_name = info[1].decode(errors='ignore')
        link_name = info[12].decode(errors='ignore')
        joint_type = type_map.get(info[2], str(info[2]))
        print(f'  {idx:>3}  {joint_name:<28}  {link_name:<24}  {joint_type}')
    print()


def make_target_transform(position: np.ndarray) -> HTM_Cls:
    # Quaternion format for HTM_Cls is [w, x, y, z]. This is a fixed downward
    # TCP orientation. IK is position-only by default, so orientation is diagnostic.
    q_down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    return HTM_Cls(None, np.float64).Rotation(q_down.tolist(), 'QUATERNION').Translation(position.tolist())


def parse_args():
    parser = argparse.ArgumentParser(description='Test ARM IK on all configuration-space vertices.')
    parser.add_argument('--c-type', choices=['Search', 'Target'], default=CONST_C_TYPE)
    parser.add_argument('--env-id', type=int, choices=[0, 1], default=0)
    parser.add_argument('--headless', action='store_true', help='Run with PyBullet DIRECT instead of GUI.')
    parser.add_argument('--no-ghost', action='store_true', help='Hide IK ghost robot.')
    parser.add_argument('--sleep', type=float, default=0.5)
    parser.add_argument('--keep-open', action='store_true', help='Keep GUI open after the test.')
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    robot_str = CONST_ROBOT_TYPE

    properties = dict(CONST_PYBULLET_ENV_PROPERTIES)
    properties['Enable_GUI'] = properties['Enable_GUI'] and not args.headless
    properties['Env_ID'] = args.env_id

    robot = PyBullet.Core.Robot_Cls(
        robot_str,
        f'{CONST_PROJECT_FOLDER}/URDFs/Robots/{robot_str.Name}/{robot_str.Name}.urdf',
        properties,
    )
    robot.Reset('Home')

    robot_id = robot._Robot_Cls__robot_id
    print_link_table(robot_id)

    vertices = robot.Get_Configuration_Space_Vertices(args.c_type)
    n_vertices = vertices.shape[0]
    print(f'Robot: {robot_str.Name}')
    print(f'TCP link: {robot_str.TCP_Link_Name}')
    print(f'{args.c_type} C-space bounds (x, y, z):')
    print(f'  min = {vertices.min(axis=0)}')
    print(f'  max = {vertices.max(axis=0)}')
    print(f'Testing vertices: {n_vertices}')
    print(f'IK: use_orientation={CONST_IK_PROPERTIES["use_orientation"]}, '
          f'tol={CONST_IK_PROPERTIES["ik_position_tolerance"]:.3f} m, '
          f'iters={CONST_IK_PROPERTIES["num_of_iteration"]}')
    print()

    count_ik_failed = 0
    count_passed = 0
    count_motor_failed = 0
    count_pose_mismatch = 0
    pos_errors_all = []
    pos_errors_passed = []
    pos_errors_failed = []
    failed_positions = []

    for idx, vertex in enumerate(vertices):
        target_t = make_target_transform(vertex)
        target_pos = target_t.p.all()

        if properties['Enable_GUI']:
            robot.Remove_External_Object('T_EE_Vertex_Viewpoint')
            robot.Add_External_Object(
                f'{CONST_PROJECT_FOLDER}/URDFs/Viewpoint/Viewpoint.urdf',
                'T_EE_Vertex_Viewpoint',
                target_t,
                None,
                0.3,
                False,
            )

        successful, theta = robot.Get_Inverse_Kinematics_Solution(
            target_t,
            CONST_IK_PROPERTIES,
            CONST_VISIBILITY_GHOST and not args.no_ghost,
        )

        if not successful:
            print(f'[{idx:2d}/{n_vertices - 1}] [IK FAILED]  target={target_pos}  theta={theta}')
            count_ik_failed += 1
            failed_positions.append(target_pos.copy())
            time.sleep(args.sleep)
            robot.Reset('Home')
            continue

        in_position = robot.Reset('Individual', theta)
        pb.stepSimulation()

        measured_t = robot.T_EE
        measured_pos = measured_t.p.all()
        pos_error = float(np.linalg.norm(target_pos - measured_pos))
        pos_errors_all.append(pos_error)

        if pos_error > CONST_POS_TOLERANCE:
            tag = 'POSE_MISMATCH'
            count_pose_mismatch += 1
            pos_errors_failed.append(pos_error)
        elif not in_position:
            tag = 'MOTOR_FAILED'
            count_motor_failed += 1
            pos_errors_failed.append(pos_error)
        else:
            tag = 'PASS'
            count_passed += 1
            pos_errors_passed.append(pos_error)

        print(f'[{idx:2d}/{n_vertices - 1}] [{tag}]  '
              f'pos_err={pos_error:.6f} m  target={target_pos}  tcp={measured_pos}')
        print(f'  theta={theta}')

        time.sleep(args.sleep)
        robot.Reset('Home')

    total = n_vertices
    print()
    print('=' * 60)
    print('=== ARM VERTEX TEST SUMMARY ===')
    print('=' * 60)
    print(f'  Total vertices           : {total}')
    print(f'  IK FAILED                : {count_ik_failed}')
    print(f'  PASS                     : {count_passed}')
    print(f'  MOTOR_FAILED             : {count_motor_failed}')
    print(f'  POSE_MISMATCH            : {count_pose_mismatch}')
    print(f'  Counter sum              : {count_ik_failed + count_passed + count_motor_failed + count_pose_mismatch}  (should == {total})')
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

    if properties['Enable_GUI'] and args.keep_open:
        print()
        print('Test complete. GUI remains open. Press Ctrl+C to exit.')
        try:
            while robot.is_connected:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    robot.Disconnect()


if __name__ == '__main__':
    sys.exit(main())
