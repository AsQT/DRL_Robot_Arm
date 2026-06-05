# System (Default)
import sys
import os
#   Resolve project root relative to this script file, then add to sys.path.
#   Works regardless of whether the script is run from its own directory,
#   from the project root, or from anywhere else.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_script_dir, '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
# Time (Time access and conversions)
import time
# Numpy (Array computing)
import numpy as np
# Custom Lib.:
#   Robotics Library for Everyone (RoLE)
#       ../RoLE/Parameters/Robot
import RoLE.Parameters.Robot as Parameters
#   PyBullet
#       ../PyBullet/Core
import PyBullet.Core
import pybullet as pb
#       ../RoLE/Transformation/Core
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
from config_loader import PROJECT_ROOT

# =============================================================================
# Constants
# =============================================================================
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.YASKAWA_GP7_Str

# Locate the path to the project folder.
CONST_PROJECT_FOLDER = str(PROJECT_ROOT)

# The properties of the PyBullet environment.
CONST_PYBULLET_ENV_PROPERTIES = {
    'Enable_GUI': True, 'fps': 100,
    'External_Base': None, 'Env_ID': 1,
    'Camera': {'Yaw': 70.0, 'Pitch': -32.0, 'Distance': 1.3,
               'Position': [0.05, -0.10, 0.06]}
}

# Configuration space type: 'Target' (validated vertices) or 'Search' (larger space).
CONST_C_TYPE = 'Target'

# Number of random samples to test in one run.  Set to a large number for a
# thorough test, or a small number for a quick sanity check.
CONST_NUM_SAMPLES = 50

# IK properties matching GP7ReachPyBulletEnv (position-only IK, 1 cm tolerance).
# The random test samples targets from the same target C-space used by the
# environment, so position-only IK should pass the vast majority of samples.
CONST_IK_PROPERTIES = {
    'delta_time': 0.01,
    'num_of_iteration': 500,         # sufficient for convergence
    'tolerance': 1e-6,
    'use_orientation': True,         # position-only IK (no orientation constraint)
    'ik_position_tolerance': 0.01,   # 1 cm
}

# Ghost visibility during the random test.
CONST_VISIBILITY_GHOST = True

# Pass/fail position error threshold (metres).  Must match ik_position_tolerance.
CONST_PASS_TOLERANCE = CONST_IK_PROPERTIES['ik_position_tolerance']


# =============================================================================
# Helper Functions
# =============================================================================
def print_error_stats(title: str, errors: list, unit: str = 'm'):
    """Print min / max / mean / std statistics for a list of scalar errors."""
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


# =============================================================================
# Main
# =============================================================================
def main():
    """
    Test random target reaching using position-only IK.

    This replicates the environment's IK behaviour so that evaluation results
    are representative of what the RL agent will encounter during training.
    """
    Robot_Str = CONST_ROBOT_TYPE

    # Create PyBullet robot.
    PyBullet_Robot_Cls = PyBullet.Core.Robot_Cls(
        Robot_Str,
        f'{CONST_PROJECT_FOLDER}/URDFs/Robots/{Robot_Str.Name}/{Robot_Str.Name}.urdf',
        CONST_PYBULLET_ENV_PROPERTIES
    )

    # Visual environment (table, etc.) is added once here.
    PyBullet_Robot_Cls.Add_Environment()

    # Reset robot to Home.
    PyBullet_Robot_Cls.Reset('Home')

    # --- Print workspace bounds ---
    target_vertices = PyBullet_Robot_Cls.Get_Configuration_Space_Vertices(CONST_C_TYPE)
    print(f'Target C-space bounds (x, y, z):')
    print(f'  min = {target_vertices.min(axis=0)}')
    print(f'  max = {target_vertices.max(axis=0)}')
    print(f'Random samples: {CONST_NUM_SAMPLES}')
    print(f'IK: use_orientation={CONST_IK_PROPERTIES["use_orientation"]}, '
          f'tol={CONST_IK_PROPERTIES["ik_position_tolerance"]:.3f} m, '
          f'iters={CONST_IK_PROPERTIES["num_of_iteration"]}, '
          f'residual={CONST_IK_PROPERTIES["tolerance"]}')
    print()

    # --- Counters ---
    count_ik_failed     = 0
    count_ik_passed    = 0   # ghost OK + motor converged
    count_motor_failed = 0   # ghost OK + motor did not converge
    count_pose_mismatch = 0  # ghost OK + real TCP > tolerance after exec

    # Position error tracking
    pos_errors_all_executed = []  # PASS + MOTOR_FAILED + POSE_MISMATCH
    pos_errors_passed      = []
    pos_errors_failed      = []   # MOTOR_FAILED + POSE_MISMATCH

    # Orientation error tracking (diagnostic only — no orientation constraint in IK)
    orient_errors_all_executed = []
    orient_errors_passed      = []
    orient_errors_failed      = []

    ik_failed_positions = []  # target positions that IK could not validate

    for sample in range(CONST_NUM_SAMPLES):
        # Generate a random target from the target C-space (same as environment).
        T_rand = PyBullet_Robot_Cls.Generate_Random_T_EE(CONST_C_TYPE, True)

        target_pos  = T_rand.p.all()
        target_quat = T_rand.Get_Rotation('QUATERNION').all()

        # --- IK solve (position-only, ghost-TCP validated) ---
        (successful, theta) = PyBullet_Robot_Cls.Get_Inverse_Kinematics_Solution(
            T_rand, CONST_IK_PROPERTIES, CONST_VISIBILITY_GHOST
        )

        # --- IK FAILED: skip real-robot execution ---
        if not successful:
            print(f'[{sample:4d}/{CONST_NUM_SAMPLES}] [IK FAILED]  '
                  f'target={target_pos}  theta={theta}')
            count_ik_failed += 1
            ik_failed_positions.append(target_pos.copy())
            time.sleep(0.3)
            PyBullet_Robot_Cls.Reset('Home')
            continue

        # --- IK SUCCEEDED: execute on real robot ---
        # in_position = PyBullet_Robot_Cls.Set_Absolute_Joint_Position(
        #     theta, {'force': 100.0, 't_0': None, 't_1': None}
        # )

        in_position = PyBullet_Robot_Cls.Reset('Individual', theta)
        pb.stepSimulation()

        T_measured        = PyBullet_Robot_Cls.T_EE
        tcp_pos_measured  = T_measured.p.all()
        tcp_quat_measured = T_measured.Get_Rotation('QUATERNION').all()
        pos_error = float(np.linalg.norm(target_pos - tcp_pos_measured))

        # Orientation is diagnostic only — position-only IK has no orientation constraint.
        q_dot     = float(np.clip(np.dot(target_quat, tcp_quat_measured), -1.0, 1.0))
        angle_deg = float(np.rad2deg(2.0 * np.arccos(q_dot)))

        # --- Classify result ---
        if pos_error > CONST_PASS_TOLERANCE:
            result_tag = 'POSE_MISMATCH'
            count_pose_mismatch += 1
            pos_errors_failed.append(pos_error)
            orient_errors_failed.append(angle_deg)
        elif not in_position:
            result_tag = 'MOTOR_FAILED'
            count_motor_failed += 1
            pos_errors_failed.append(pos_error)
            orient_errors_failed.append(angle_deg)
        else:
            result_tag = 'PASS'
            count_ik_passed += 1
            pos_errors_passed.append(pos_error)
            orient_errors_passed.append(angle_deg)

        # All executed samples (IK succeeded + robot moved)
        pos_errors_all_executed.append(pos_error)
        orient_errors_all_executed.append(angle_deg)

        print(f'[{sample:4d}/{CONST_NUM_SAMPLES}] [{result_tag}]  '
              f'pos_err={pos_error:.6f} m  orient_err={angle_deg:.2f} deg  '
              f'target={target_pos}')

        time.sleep(0.3)
        PyBullet_Robot_Cls.Reset('Home')

    # --- Summary ---
    total = CONST_NUM_SAMPLES
    print()
    print('=' * 60)
    print('=== RANDOM TEST SUMMARY ===')
    print('=' * 60)
    print(f'  Total samples            : {total}')
    print(f'  IK FAILED                : {count_ik_failed}   (ghost TCP > {CONST_PASS_TOLERANCE:.3f} m from target)')
    print(f'  PASS                     : {count_ik_passed}    (ghost OK + motor converged)')
    print(f'  MOTOR_FAILED             : {count_motor_failed}    (ghost OK + motor failed)')
    print(f'  POSE_MISMATCH            : {count_pose_mismatch}    (ghost OK + real TCP > {CONST_PASS_TOLERANCE:.3f} m)')
    print(f'  IK succeeded total         : {count_ik_passed + count_motor_failed + count_pose_mismatch}')

    counter_sum = count_ik_failed + count_ik_passed + count_motor_failed + count_pose_mismatch
    print(f'  Counter sum               : {counter_sum}  (should == {total})')

    # --- Position error statistics ---
    print()
    print('--- Position Error Statistics ---')
    print_error_stats(
        'ALL EXECUTED position errors (PASS + MOTOR_FAILED + POSE_MISMATCH)',
        pos_errors_all_executed, 'm'
    )
    print_error_stats('PASS position errors', pos_errors_passed, 'm')
    print_error_stats(
        'FAILED executed position errors (MOTOR_FAILED + POSE_MISMATCH)',
        pos_errors_failed, 'm'
    )

    # --- Orientation error statistics ---
    print()
    print('--- Orientation Error Statistics ---')
    print_error_stats(
        'ALL EXECUTED orientation errors (PASS + MOTOR_FAILED + POSE_MISMATCH)',
        orient_errors_all_executed, 'deg'
    )
    print_error_stats('PASS orientation errors', orient_errors_passed, 'deg')
    print_error_stats(
        'FAILED executed orientation errors (MOTOR_FAILED + POSE_MISMATCH)',
        orient_errors_failed, 'deg'
    )

    # --- IK-FAILED target positions ---
    if ik_failed_positions:
        arr = np.array(ik_failed_positions)
        print()
        print(f'  IK-FAILED target positions (x, y, z):')
        print(f'    count = {len(ik_failed_positions)}')
        print(f'    min   = {arr.min(axis=0)}')
        print(f'    max   = {arr.max(axis=0)}')
        print(f'    mean  = {arr.mean(axis=0)}')
    else:
        print()
        print(f'  IK-FAILED target positions: none')

    if counter_sum != total:
        print()
        print(f'  [BUG] Counter mismatch!')

    # Keep GUI open for inspection.
    print()
    print('Test complete. GUI remains open. Press Ctrl+C to exit.')
    try:
        while PyBullet_Robot_Cls.is_connected:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    PyBullet_Robot_Cls.Disconnect()


if __name__ == '__main__':
    sys.exit(main())
