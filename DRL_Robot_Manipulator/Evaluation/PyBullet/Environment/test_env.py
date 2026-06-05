# System (Default)
import sys
import os

# Resolve the src/ directory relative to this script file.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# OS (Operating system interfaces)
import os as _os_alias  # noqa: F401
# Custom Lib.:
import RoLE.Parameters.Robot as Parameters
import PyBullet.Core
from config_loader import PROJECT_FOLDER_NAME

# Locate the path to the project folder.
CONST_PROJECT_FOLDER = os.getcwd().split(PROJECT_FOLDER_NAME)[0] + PROJECT_FOLDER_NAME

# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.YASKAWA_GP7_Str

# The properties of the PyBullet environment.
CONST_PYBULLET_ENV_PROPERTIES = {
    'Enable_GUI': True, 'fps': 100,
    'External_Base': None,
    'Env_ID': 0,
    'Camera': {'Yaw': 70.0, 'Pitch': -32.0, 'Distance': 1.3,
               'Position': [0.05, -0.10, 0.06]}
}


def main():
    Robot_Str = CONST_ROBOT_TYPE

    PyBullet_Robot_Cls = PyBullet.Core.Robot_Cls(
        Robot_Str,
        f'{CONST_PROJECT_FOLDER}/URDFs/Robots/{Robot_Str.Name}/{Robot_Str.Name}.urdf',
        CONST_PYBULLET_ENV_PROPERTIES)

    PyBullet_Robot_Cls.Reset('Home')

    PyBullet_Robot_Cls.Add_External_Object(
        f'{CONST_PROJECT_FOLDER}/URDFs/Viewpoint/Viewpoint.urdf',
        'T_EE_Viewpoint', PyBullet_Robot_Cls.T_EE, None,
        0.3, False)

    while PyBullet_Robot_Cls.is_connected:
        PyBullet_Robot_Cls.Step()

    PyBullet_Robot_Cls.Disconnect()


if __name__ == '__main__':
    sys.exit(main())
