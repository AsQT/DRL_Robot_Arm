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
# Numpy (Arline computing) [pip3 install numpy]
import numpy as np
# Time (Time access and conversions)
import time
# Custom Lib.:
#   Robotics Library for Everyone (RoLE)
#       ../RoLE/Parameters/Robot
import RoLE.Parameters.Robot as Parameters
#       ../RoLE/Transformation/Core
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
#   PyBullet
#       ../PyBullet/Core
import PyBullet.Core
import pybullet as pb
from config_loader import PROJECT_ROOT

# =============================================================================
# Constants
# =============================================================================
CONST_ROBOT_TYPE = Parameters.YASKAWA_GP7_Str
CONST_PROJECT_FOLDER = str(PROJECT_ROOT)
CONST_PYBULLET_ENV_PROPERTIES = {
    'Enable_GUI': True, 'fps': 100,
    'External_Base': None, 'Env_ID': 0,
    'Camera': {'Yaw': 70.0, 'Pitch': -32.0, 'Distance': 1.3,
               'Position': [0.05, -0.10, 0.06]}
}
CONST_C_TYPE = 'Target'
CONST_IK_PROPERTIES = {
    'delta_time': 0.1, 'num_of_iteration': 500,
    'tolerance': 1e-30
}
# Environment-style props matching GP7ReachPyBulletEnv defaults.
# Position-only IK + 1 cm ghost-TCP tolerance.
CONST_ENV_IK_PROPERTIES = {
    'delta_time': 0.1, 'num_of_iteration': 500,
    'tolerance': 1e-30,
    'use_orientation': False,
    'ik_position_tolerance': 0.01,
}
CONST_VISIBILITY_GHOST = True
CONST_POS_TOLERANCE = 0.010  # metres


# =============================================================================
# Diagnostic helpers (do NOT change production Robot_Cls)
# =============================================================================

def _print_link_table(robot_id):
    """Print a table of all PyBullet joint/link indices for the robot."""
    print('=== PYBULLET LINK/JOINT TABLE ===')
    print(f'  {"idx":>3}  {"joint_name":<35}  {"link_name":<20}  {"joint_type"}')
    print('  ' + '-' * 80)
    num_joints = pb.getNumJoints(robot_id)
    for idx in range(num_joints):
        info = pb.getJointInfo(robot_id, idx)
        joint_name = info[1].decode('utf-8') if isinstance(info[1], bytes) else info[1]
        link_name  = info[12].decode('utf-8') if isinstance(info[12], bytes) else info[12]
        jtype_map = {0: 'REVOLUTE', 1: 'PRISMATIC', 2: 'SPHERICAL', 3: 'PLANAR', 4: 'FIXED'}
        jtype = jtype_map.get(info[2], str(info[2]))
        # Also flag the base (-1)
        if idx == 0:
            base_pos, base_quat = pb.getBasePositionAndOrientation(robot_id)
            print(f'  {"-1":>3}  {"BASE":<35}  {"base_link":<20}  {"FIXED"}')
        print(f'  {idx:>3}  {joint_name:<35}  {link_name:<20}  {jtype}')
    print()


def _get_ee_link_index(robot_id):
    """Return the PyBullet link index of the ee_link if present, else None."""
    num_joints = pb.getNumJoints(robot_id)
    for idx in range(num_joints):
        info = pb.getJointInfo(robot_id, idx)
        link_name = info[12].decode('utf-8') if isinstance(info[12], bytes) else info[12]
        if 'ee_link' in link_name.lower() or 'end_effector' in link_name.lower():
            return idx
    return None


def _get_link_index_by_name(robot_id, expected_link_name):
    """Return the PyBullet link index with an exact child-link name match."""
    num_joints = pb.getNumJoints(robot_id)
    for idx in range(num_joints):
        info = pb.getJointInfo(robot_id, idx)
        link_name = info[12].decode('utf-8') if isinstance(info[12], bytes) else info[12]
        if link_name == expected_link_name:
            return idx
    return None


def _get_link_name(robot_id, link_index):
    """Return child-link name for a PyBullet link index, or None if invalid."""
    if link_index is None:
        return None
    info = pb.getJointInfo(robot_id, link_index)
    return info[12].decode('utf-8') if isinstance(info[12], bytes) else info[12]


def _call_ik_6d(robot_id, link_index, target_position, target_orientation_xyzw):
    """Call PyBullet 6-D IK (position + orientation)."""
    return pb.calculateInverseKinematics(
        robot_id, link_index,
        target_position.tolist(),
        targetOrientation=target_orientation_xyzw.tolist()
    )


def _call_ik_position_only(robot_id, link_index, target_position):
    """Call PyBullet position-only IK (no orientation)."""
    return pb.calculateInverseKinematics(
        robot_id, link_index,
        target_position.tolist()
    )


def _validate_ik_on_ghost(robot_id_ghost, theta_index, theta,
                           link_index, target_position, tolerance):
    """
    Reset ghost robot to theta, read back TCP, return (successful, pos_error, ghost_tcp_pos).
    """
    for idx, th_index in enumerate(theta_index):
        pb.resetJointState(robot_id_ghost, th_index, theta[idx])
    link_state = pb.getLinkState(robot_id_ghost, link_index, computeForwardKinematics=True)
    ghost_tcp_pos = np.array(link_state[4])
    pos_error = float(np.linalg.norm(target_position - ghost_tcp_pos))
    successful = (pos_error <= tolerance)
    return successful, pos_error, ghost_tcp_pos


def _run_ik_diagnostic(robot_id, robot_id_ghost, theta_index,
                        link_index, link_name,
                        C_vertices, q_Down, use_6d_ik):
    """
    Run IK diagnostic for all vertices using a given link index and IK mode.

    Returns:
        results: list of dicts with keys: vertex_idx, target_pos, target_quat,
                  theta, pos_error, successful
    """
    results = []
    for vi in range(C_vertices.shape[0]):
        target_pos = C_vertices[vi]
        T_v = HTM_Cls(None, np.float64).Rotation(q_Down.tolist(), 'QUATERNION').Translation(target_pos.tolist())
        target_quat_wxyz = T_v.Get_Rotation('QUATERNION')  # [w, x, y, z]
        target_orientation_xyzw = np.array([
            target_quat_wxyz.x, target_quat_wxyz.y,
            target_quat_wxyz.z, target_quat_wxyz.w
        ])  # [x, y, z, w] for PyBullet

        # Call IK
        if use_6d_ik:
            theta_raw = _call_ik_6d(robot_id, link_index, target_pos, target_orientation_xyzw)
        else:
            theta_raw = _call_ik_position_only(robot_id, link_index, target_pos)

        theta = np.array(theta_raw[:len(theta_index)], dtype=np.float64)

        # Ghost validation
        successful, pos_error, ghost_tcp_pos = _validate_ik_on_ghost(
            robot_id_ghost, theta_index, theta,
            link_index, target_pos, CONST_POS_TOLERANCE
        )
        results.append({
            'vertex_idx': vi,
            'target_pos': target_pos,
            'target_quat': target_quat_wxyz.all(),
            'ghost_tcp_pos': ghost_tcp_pos,
            'theta': theta,
            'pos_error': pos_error,
            'successful': successful,
        })
    return results


def _print_results_table(mode_label, link_index, link_name, results):
    """Print per-vertex results and a compact summary for one IK configuration."""
    n = len(results)
    passed = [r for r in results if r['successful']]
    failed = [r for r in results if not r['successful']]

    if passed:
        pos_errors_passed = [r['pos_error'] for r in passed]
        max_pos_err = max(pos_errors_passed)
        mean_pos_err = float(np.mean(pos_errors_passed))
    else:
        max_pos_err = float('nan')
        mean_pos_err = float('nan')

    print(f'--- {mode_label}  |  link idx={link_index} ({link_name}) ---')
    for r in results:
        tag = 'PASS' if r['successful'] else 'FAIL'
        print(f'  v{r["vertex_idx"]}: {tag}  pos_err={r["pos_error"]:.6f} m  '
              f'target={r["target_pos"]}')

    print(f'  SUMMARY: passed={len(passed)}/{n}  '
          f'max_pos_err={max_pos_err:.6f} m  mean_pos_err={mean_pos_err:.6f} m')
    print()
    return {
        'mode': mode_label,
        'link_index': link_index,
        'link_name': link_name,
        'passed': len(passed),
        'failed': len(failed),
        'total': n,
        'max_pos_error': max_pos_err,
        'mean_pos_error': mean_pos_err,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    Robot_Str = CONST_ROBOT_TYPE

    # --- Create PyBullet robot ---
    PyBullet_Robot_Cls = PyBullet.Core.Robot_Cls(
        Robot_Str,
        f'{CONST_PROJECT_FOLDER}/URDFs/Robots/{Robot_Str.Name}/{Robot_Str.Name}.urdf',
        CONST_PYBULLET_ENV_PROPERTIES
    )
    robot_id        = PyBullet_Robot_Cls._Robot_Cls__robot_id
    robot_id_ghost  = PyBullet_Robot_Cls._Robot_Cls__robot_id_ghost
    theta_index     = PyBullet_Robot_Cls._Robot_Cls__theta_index

    PyBullet_Robot_Cls.Reset('Home')

    # --- Print link table once ---
    _print_link_table(robot_id)

    ee_link_idx = _get_ee_link_index(robot_id)
    ee_link_name = _get_link_name(robot_id, ee_link_idx)
    link_6_idx = _get_link_index_by_name(robot_id, 'link_6')
    link_6_name = _get_link_name(robot_id, link_6_idx)
    print(f'ee_link detected at PyBullet index: {ee_link_idx}')
    if ee_link_idx is not None:
        print(f'  name: {ee_link_name}')
    print(f'link_6 detected at PyBullet index: {link_6_idx}')
    print()

    # --- Get vertices ---
    C_vertices = PyBullet_Robot_Cls.Get_Configuration_Space_Vertices(CONST_C_TYPE)
    n_vertices = C_vertices.shape[0]
    print(f'Testing {n_vertices} target-space vertices\n')

    # Fixed downward orientation for all targets.
    q_Down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # [w, x, y, z]

    # --- Production: 6-D IK targeting ee_link (the true TCP) ---
    if ee_link_idx is not None:
        print('=' * 60)
        print(f'PRODUCTION: 6-D IK targeting ee_link (index={ee_link_idx})')
        print('=' * 60)
        prod_results = _run_ik_diagnostic(
            robot_id, robot_id_ghost, theta_index,
            link_index=ee_link_idx, link_name=ee_link_name,
            C_vertices=C_vertices, q_Down=q_Down, use_6d_ik=True
        )
        prod_summary = _print_results_table(f'6D IK (ee_link #{ee_link_idx})', ee_link_idx, ee_link_name, prod_results)
    else:
        print('[WARN] ee_link not found - cannot run production 6-D IK test')
        prod_summary = None

    # --- Diagnostic: link_6 (old TCP, before tool/ee chain was added to URDF) ---
    diag_a_summary = None
    if link_6_idx is not None:
        print('=' * 60)
        print(f'DIAGNOSTIC A: Position-only IK, link_index={link_6_idx} (link_6 / old TCP)')
        print('=' * 60)
        diag_a_results = _run_ik_diagnostic(
            robot_id, robot_id_ghost, theta_index,
            link_index=link_6_idx, link_name=link_6_name,
            C_vertices=C_vertices, q_Down=q_Down, use_6d_ik=False
        )
        diag_a_summary = _print_results_table('Pos-only IK (link_6 / OLD)', link_6_idx, link_6_name, diag_a_results)
    else:
        print('[WARN] link_6 not found - skipping diagnostic A.\n')

    # --- Diagnostic B/C: if ee_link exists, test it ---
    diag_b_summary = None
    if ee_link_idx is not None:
        print('=' * 60)
        print(f'DIAGNOSTIC B: 6-D IK, link_index={ee_link_idx} ({ee_link_name})')
        print('=' * 60)
        diag_b6d_results = _run_ik_diagnostic(
            robot_id, robot_id_ghost, theta_index,
            link_index=ee_link_idx, link_name=ee_link_name,
            C_vertices=C_vertices, q_Down=q_Down, use_6d_ik=True
        )
        diag_b_summary = _print_results_table(f'6D IK', ee_link_idx, ee_link_name, diag_b6d_results)

        print('=' * 60)
        print(f'DIAGNOSTIC C: Position-only IK, link_index={ee_link_idx} ({ee_link_name})')
        print('=' * 60)
        diag_c_results = _run_ik_diagnostic(
            robot_id, robot_id_ghost, theta_index,
            link_index=ee_link_idx, link_name=ee_link_name,
            C_vertices=C_vertices, q_Down=q_Down, use_6d_ik=False
        )
        diag_c_summary = _print_results_table(f'Pos-only IK', ee_link_idx, ee_link_name, diag_c_results)
    else:
        print('ee_link not found in URDF - skipping diagnostics B and C.\n')

    # --- Environment-style test: position-only IK targeting ee_link (matches GP7ReachPyBulletEnv) ---
    env_style_summary = None
    if ee_link_idx is not None:
        print('=' * 60)
        print(f'ENVIRONMENT STYLE: Position-only IK, ee_link (index={ee_link_idx}), tol=0.01 m')
        print('=' * 60)
        env_style_results = _run_ik_diagnostic(
            robot_id, robot_id_ghost, theta_index,
            link_index=ee_link_idx, link_name=ee_link_name,
            C_vertices=C_vertices, q_Down=q_Down, use_6d_ik=False
        )
        env_style_tol = 0.01
        for r in env_style_results:
            r['successful'] = (r['pos_error'] <= env_style_tol)
        env_style_summary = _print_results_table(
            f'Pos-only IK (ee_link #{ee_link_idx}, tol={env_style_tol})', ee_link_idx, ee_link_name, env_style_results
        )
    else:
        print('[WARN] ee_link not found - skipping environment-style diagnostic.\n')

    # --- Compact comparison table ---
    print('=' * 60)
    print('COMPARISON TABLE')
    print('=' * 60)
    print('  Note: TCP = ee_link (resolved at runtime). link_6 (index 5) is shown as old diagnostic.')
    print(f'  TCP link name = ee_link  |  TCP link index = {ee_link_idx}')
    print()
    print(f'  {"IK mode":<45}  {"link_idx":<10}  {"link_name":<15}  '
          f'{"passed":>6}  {"failed":>6}  {"max_pos_err(m)":>14}  {"mean_pos_err(m)":>15}')
    print('  ' + '-' * 125)
    all_summaries = []
    if prod_summary is not None:
        all_summaries.append(prod_summary)
    if diag_a_summary is not None:
        all_summaries.append(diag_a_summary)
    if diag_b_summary is not None:
        all_summaries.append(diag_b_summary)
    if diag_c_summary is not None:
        all_summaries.append(diag_c_summary)
    if env_style_summary is not None:
        all_summaries.append(env_style_summary)
    for s in all_summaries:
        print(f'  {s["mode"]:<45}  {s["link_index"]:>10}  {s["link_name"]:<15}  '
              f'{s["passed"]:>6}  {s["failed"]:>6}  '
              f'{s["max_pos_error"]:>14.6f}  {s["mean_pos_error"]:>15.6f}')

    # --- Production run: single clean pass - environment-style position-only IK ---
    print()
    print('=' * 60)
    print('PRODUCTION RUN: Robot_Cls using environment-style position-only IK')
    print('=' * 60)
    count_ik_failed     = 0
    count_ik_passed     = 0
    count_motor_failed  = 0
    count_pose_mismatch  = 0
    max_pos_err_passed   = 0.0

    for i in range(n_vertices):
        q_Down_i = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        T_vertex = HTM_Cls(None, np.float64).Rotation(q_Down_i.tolist(), 'QUATERNION').Translation(C_vertices[i].tolist())

        # Update ghost target frame visualiser
        PyBullet_Robot_Cls.Remove_External_Object('T_EE_Rand_Viewpoint')
        PyBullet_Robot_Cls.Add_External_Object(
            f'{CONST_PROJECT_FOLDER}/URDFs/Viewpoint/Viewpoint.urdf',
            'T_EE_Rand_Viewpoint', T_vertex, None, 0.3, False
        )

        (successful, theta) = PyBullet_Robot_Cls.Get_Inverse_Kinematics_Solution(
            T_vertex, CONST_ENV_IK_PROPERTIES, CONST_VISIBILITY_GHOST
        )
        target_pos  = T_vertex.p.all()
        target_quat = T_vertex.Get_Rotation('QUATERNION').all()

        if not successful:
            print(f'=== VERTEX {i}/{n_vertices-1}  [IK FAILED] ===')
            print(f'  Target position = {target_pos}')
            print(f'  Target quaternion [w,x,y,z] = {target_quat}')
            print(f'  Theta from IK (not executed): {theta}')
            count_ik_failed += 1
        else:
            in_position = PyBullet_Robot_Cls.Set_Absolute_Joint_Position(
                theta, {'force': 100.0, 't_0': None, 't_1': None}
            )
            T_measured        = PyBullet_Robot_Cls.T_EE
            tcp_pos_measured  = T_measured.p.all()
            tcp_quat_measured = T_measured.Get_Rotation('QUATERNION').all()
            pos_error = float(np.linalg.norm(target_pos - tcp_pos_measured))

            q_dot     = np.clip(np.dot(target_quat, tcp_quat_measured), -1.0, 1.0)
            angle_deg = float(np.rad2deg(2.0 * np.arccos(q_dot)))

            if pos_error > CONST_ENV_IK_PROPERTIES['ik_position_tolerance']:
                result_tag = 'POSE_MISMATCH'
                count_pose_mismatch += 1
            elif not in_position:
                result_tag = 'MOTOR_FAILED'
                count_motor_failed += 1
            else:
                result_tag = 'PASS'
                count_ik_passed += 1
                max_pos_err_passed = max(max_pos_err_passed, pos_error)

            print(f'=== VERTEX {i}/{n_vertices-1}  [{result_tag}] ===')
            print(f'  Target position = {target_pos}')
            print(f'  TCP position   = {tcp_pos_measured}')
            print(f'  Position error = {pos_error:.6f} m  '
                  f'(threshold: {CONST_ENV_IK_PROPERTIES["ik_position_tolerance"]:.3f} m)')
            print(f'  Target quaternion [w,x,y,z] = {target_quat}  (diagnostic only)')
            print(f'  TCP quaternion [w,x,y,z]     = {tcp_quat_measured}')
            print(f'  Orientation error = {angle_deg:.4f} deg  (diagnostic only)')
            print(f'  Commanded theta  = {theta}')
            print(f'  Actual theta     = {PyBullet_Robot_Cls.Theta}')
            print(f'  successful={successful}, in_position={in_position}')

        time.sleep(0.5)
        PyBullet_Robot_Cls.Reset('Home')

    print()
    print('=== PRODUCTION RUN SUMMARY ===')
    total = n_vertices
    print(f'  Total vertices           : {total}')
    print(f'  IK FAILED                : {count_ik_failed}')
    print(f'  PASS                     : {count_ik_passed}   (ghost OK + motor converged)')
    print(f'  MOTOR_FAILED             : {count_motor_failed}  (ghost OK + motor failed)')
    print(f'  POSE_MISMATCH            : {count_pose_mismatch}  '
          f'(ghost OK, real TCP > {CONST_ENV_IK_PROPERTIES["ik_position_tolerance"]:.3f} m)')
    print(f'  IK succeeded total        : {count_ik_passed + count_motor_failed + count_pose_mismatch}')
    if count_ik_passed > 0:
        print(f'  Max position error (PASS): {max_pos_err_passed:.6f} m')

    total_check = count_ik_failed + count_ik_passed + count_motor_failed + count_pose_mismatch
    print(f'  Counter sum              : {total_check}  (should == {total})')
    if total_check != total:
        print('  [BUG] Counter mismatch detected!')

    # Keep GUI open after test
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
