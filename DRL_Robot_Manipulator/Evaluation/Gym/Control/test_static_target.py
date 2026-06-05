# System (Default)
import sys
import os

# Find the project root by searching upward for the 'src' directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = None
for _parent in (_SCRIPT_DIR, None):
    if _parent is None:
        break
    _check = os.path.join(_parent, 'src')
    if os.path.isdir(_check):
        _SRC_DIR = _check
        break
    _next = os.path.dirname(_parent)
    if _next == _parent:
        break
    _parent = _next
if _SRC_DIR is None:
    # Fallback: assume project root is cwd
    _SRC_DIR = os.path.join(os.getcwd(), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Numpy (Array computing)
import numpy as np
# Custom Lib.:
import RoLE.Parameters.Robot as Parameters
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
import RoLE.Utilities.File_IO
import RoLE.Kinematics.Core
import PyBullet.Utilities
import PyBullet.Core
from config_loader import PROJECT_FOLDER_NAME

# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.YASKAWA_GP7_Str
CONST_IK_PROPERTIES = {'delta_time': 0.1, 'num_of_iteration': 500, 'tolerance': 1e-30}
CONST_ENV_MODE = 'Default'
CONST_PYBULLET_ENV_PROPERTIES = {
    'Enable_GUI': True, 'fps': 100,
    'External_Base': None,
    'Env_ID': 0 if CONST_ENV_MODE == 'Default' else 1,
    'Camera': {'Yaw': 70.0, 'Pitch': -32.0, 'Distance': 1.3,
               'Position': [0.05, -0.10, 0.06]}
}
CONST_ALGORITHM = 'DDPG'
CONST_PROJECT_FOLDER = os.getcwd().split(PROJECT_FOLDER_NAME)[0] + PROJECT_FOLDER_NAME


def main():
    Robot_Str = CONST_ROBOT_TYPE

    Env_Structure = PyBullet.Utilities.Get_Environment_Structure(
        Robot_Str.Name, 0 if CONST_ENV_MODE == 'Default' else 1)

    v = np.array([
        Env_Structure.C.Target.T.p.x + (Env_Structure.C.Target.Size[0] / 4.0),
        Env_Structure.C.Target.T.p.y + (-1) * (Env_Structure.C.Target.Size[1] / 4.0),
        Env_Structure.C.Target.T.p.z], dtype=np.float64)

    file_path = f'{CONST_PROJECT_FOLDER}/Data/Prediction/Environment_{CONST_ENV_MODE}/{CONST_ALGORITHM}/{Robot_Str.Name}/path_static_target'

    data = RoLE.Utilities.File_IO.Load(file_path, 'txt', ',')

    PyBullet_Robot_Cls = PyBullet.Core.Robot_Cls(
        Robot_Str,
        f'{CONST_PROJECT_FOLDER}/URDFs/Robots/{Robot_Str.Name}/{Robot_Str.Name}.urdf',
        CONST_PYBULLET_ENV_PROPERTIES)

    PyBullet_Robot_Cls.Reset('Home')

    q_0 = PyBullet_Robot_Cls.T_EE.Get_Rotation('QUATERNION').all()
    T = HTM_Cls(None, np.float64).Rotation(q_0, 'QUATERNION').Translation(v)
    PyBullet_Robot_Cls.Add_External_Object(
        f'{CONST_PROJECT_FOLDER}/URDFs/Viewpoint/Viewpoint.urdf', 'T_EE_Rand_Viewpoint', T,
        None, 0.3, False)
    PyBullet_Robot_Cls.Add_External_Object(
        f'{CONST_PROJECT_FOLDER}/URDFs/Primitives/Sphere/Sphere.urdf', 'T_EE_Rand_Sphere', T,
        [0.0, 1.0, 0.0, 0.2], 0.01, False)

    (_, theta_f) = RoLE.Kinematics.Core.Inverse_Kinematics_Numerical(
        T, PyBullet_Robot_Cls.Theta, 'Levenberg-Marquardt', Robot_Str,
        {'delta_time': 0.2, 'num_of_iteration': 500, 'tolerance': 1e-30})
    PyBullet_Robot_Cls.Reset('Individual', theta_f, True)

    theta_0 = PyBullet_Robot_Cls.Theta
    theta_arr = []
    for data_i in data:
        T_i = HTM_Cls(None, np.float64).Rotation(q_0, 'QUATERNION').Translation(data_i)
        (_, theta_i) = RoLE.Kinematics.Core.Inverse_Kinematics_Numerical(
            T_i, theta_0, 'Levenberg-Marquardt', Robot_Str, CONST_IK_PROPERTIES)
        theta_arr.append(theta_i)
        theta_0 = theta_i.copy()

    i = 0
    while PyBullet_Robot_Cls.is_connected:
        if i < len(theta_arr):
            _ = PyBullet_Robot_Cls.Set_Absolute_Joint_Position(
                theta_arr[i],
                {'force': 100.0, 't_0': 0.0,
                 't_1': np.round(1.0 / data[:, 0].size, 2)})
            i += 1
        else:
            PyBullet_Robot_Cls.Step()

    PyBullet_Robot_Cls.Disconnect()


if __name__ == '__main__':
    sys.exit(main())
