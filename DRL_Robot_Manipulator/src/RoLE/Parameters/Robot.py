"""
RoLE Parameters — Robot Definitions
=================================
Dataclass definitions for robot physical and kinematic parameters: Denavit-Hartenberg
tables, joint limits, home configurations, collision structures, and transform matrices.

This module exports pre-defined parameter structures for supported robot models
(e.g. ``YASKAWA_GP7_Str``) used throughout the RoLE kinematics and PyBullet layers.
"""

# Numpy (Array computing) [pip3 install numpy]tp
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Lib.: Robotics Library for Everyone (RoLE)
#   ../RoLE/Transformation/Core
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
#   ../RoLE/Transformation/Utilities/Mathematics
import RoLE.Transformation.Utilities.Mathematics as Mathematics
#   ../RoLE/Primitives/Core
from RoLE.Primitives.Core import Box_Cls
#   ../RoLE/Primitives/Core
from RoLE.Collider.Core import AABB_Cls, OBB_Cls

@dataclass
class DH_Parameters_Str:
    """
    Description:
        The auxiliary structure of the Denavit-Hartenberg (DH) parameters.

        Note:
            DH (Denavit-Hartenberg) parameters: 
                (1) theta_zero [Vector<float>]: Joint angle (Theta_i). Rotation part in radians.
                                                Unit: [radian]                        
                (2) a [Vector<float>]: Link length (a_i). Translation part in meters.
                                       Unit: [meter]
                (3) d [Vector<float>]: Link offset (d_i). Translation part in meters.
                                       Unit: [meter]
                (4) alpha [Vector<float>]: Link twist (alpha_i). Rotation part in radians.
                                           Unit: [radian]
    """

    # Standard Denavit-Hartenberg (DH):
    #       DH_theta_zero = th{i} + theta_zero{i}
    #       DH_a          = a{i}
    #       DH_d          = d{i}
    #       DH_alpha      = alpha{i}
    #   Unit [Matrix<float>]
    Standard: tp.List[tp.List[float]] = field(default_factory=list)
    # Modified Denavit-Hartenberg (DH):
    #       DH_theta_zero = th{i} + theta_zero{i}
    #       DH_a          = a{i - 1}
    #       DH_d          = d{i}
    #       DH_alpha      = alpha{i - 1}
    #   Unit [Matrix<float>]
    Modified: tp.List[tp.List[float]] = field(default_factory=list)

@dataclass
class Theta_Parameters_Str(object):
    """
    Description:
        The auxiliary structure of the joint (theta) parameters.
    """

    # Zero absolute position of each joint.
    #   Unit [Vector<float>]
    Zero: tp.List[float] = field(default_factory=list)
    # Home absolute position of each joint.
    #   Unit [Vector<float>]
    Home: tp.List[float] = field(default_factory=list)
    # Limits of absolute joint position in radians and meters.
    #   Unit [Matrix<float>]
    Limit: tp.List[tp.List[float]] = field(default_factory=list)
    # Other parameters of the object structure.
    #   The name of the joints.
    #       Unit [Vector<string>]
    Name: tp.List[str] = field(default_factory=list)
    #   Identification of the type of joints.
    #       Note: R - Revolute, P - Prismatic
    #       Unit [Vector<string>]
    Type: tp.List[str] = field(default_factory=list)
    #   Identification of the axis of the absolute position of the joint. 
    #       Note: 'X', 'Z'
    #       Unit [Vector<string>]
    Axis: tp.List[str] = field(default_factory=list)
    #   Identification of the axis direction.
    #       Note: (+1) - Positive, (-1) - Negative
    #       Unit [Vector<int>]
    Direction: tp.List[int] = field(default_factory=list)

@dataclass
class T_Parameters_Str:
    """
    Description:
        The auxiliary structure of the homogeneous transformation matrix {T} parameters.
    """

    # Homogeneous transformation matrix of the base.
    #   Unit [Matrix<float>]
    Base: tp.List[tp.List[float]] = field(default_factory=list)
    # Homogeneous transformation matrix of the end-effector (tool).
    #   Unit [Matrix<float>]
    End_Effector: tp.List[tp.List[float]] = field(default_factory=list)
    # The zero configuration of the homogeneous transformation 
    # matrix of each joint (theta). The method (Standard, Modified) chosen 
    # to determine the configuration depends on the specific task.
    #   Unit [Matrix<float>]
    Zero_Cfg: tp.List[tp.List[float]] = field(default_factory=list)

@dataclass
class Collider_Str:
    """
    Description:
        The auxiliary structure of both the internal and external colliders.

        Note:
            Internal colliders are generated from the program, see below:
                ./src/Evaluation/Blender/Collider/gen_colliders.py
    """

    # Internal colliders of the base.
    #   Unit [Tuple<OBB_Cls(object)>]
    Base: tp.Tuple[OBB_Cls] = field(default_factory=tuple)
    # Internal colliders of the joints.
    #   Unit [Tuple<OBB_Cls(object)>]
    Theta: tp.Tuple[OBB_Cls] = field(default_factory=tuple)
    # Offset of the self-collision detection function.
    #   Note:
    #       If the offset is equal to 0, the function checks all 
    #       combinations of collisions.
    #   Unit [int]
    Offset: int = 0
    # External colliders.
    #   Unit [Tuple<AABB_Cls(object)>/Tuple<OBB_Cls(object)>]
    External: tp.Tuple[tp.Union[AABB_Cls, OBB_Cls]] = field(default_factory=tuple)
    # Optimized collision pairs.
    #   Note:
    #       The script to optimize collision pairs can be found here:
    #           ../Evaluation/Kinematics/Collider/optimize_collision_pairs.py
    #   Unit [Matrix<float>]
    Pairs: tp.List[tp.List[float]] = field(default_factory=list)

@dataclass
class Robot_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the robot.

    Initialization of the Class (structure):
        Input:
            (1) Name [string]: Name of the robotic structure.
            (2) Id [int]: Identification number.

    Example:
        Initialization:
            Cls = Robot_Parameters_Str(name)
            Cls.Name = ...
            ...
            Cls.T = ..
    """

    # Name of the robotic structure.
    #   Unit [string]
    Name: str = ''
    # Identification number.
    #   Unit [int]
    Id: int = 0
    # Denavit-Hartenberg (DH) parameters.
    #   Unit [DH_Parameters_Str(object)]
    DH: DH_Parameters_Str = field(default_factory=DH_Parameters_Str)
    # Absolute joint position (theta) parameters.
    #   Unit [Theta_Parameters_Str(object)]
    Theta: Theta_Parameters_Str = field(default_factory=Theta_Parameters_Str)
    # Homogeneous transformation matrix (T) parameters.
    #   Unit [T_Parameters_Str(object)]
    T: T_Parameters_Str = field(default_factory=T_Parameters_Str)
    # Internal and external colliders of the robot structure.
    #   Unit [Collider_Str(object)]
    Collider: Collider_Str = field(default_factory=Collider_Str)
    # Information about whether the external axis is part of the robot 
    # or not. For example, a linear track.
    #   Unit [bool]
    External_Axis: bool = False

"""
Robot Type - YASKAWA GP7:
    Absolute Joint Position:
        Joint 1: [+/- 170.0] [°]
        Joint 2: [- 65.0, +145] [°]
        Joint 3: [-116.0, +255.0] [°]
        Joint 4: [+/- 190.0] [°]
        Joint 5: [+/- 135.0] [°]
        Joint 6: [+/- 360.0] [°]

    Denavit-Hartenberg (DH) Standard:
        theta_zero = [  0.0,    -1.57,       0,     0,   0,   0]
        a          = [  0.0,    0.040,   0.445,     0.040,   0.0,   0.0]
        d          = [  0.330,    0.0,   0.0,     0.440,   0.0, 0.080]
        alpha      = [  0.0,      -1.57,  0.0,     -1.57,   1.57,  -1.57]
    Denavit-Hartenberg (DH) modified:
        theta_zero = [  0.0,    -1.57,   0.0,   0.0,     0,     0.0]
        a          = [  0.0,   0.040,  0.455,   0.040,     0.0,   0.0]
        d          = [  0.330,     0.0,   0.0,  0.440,   0.0, 0.080]
        alpha      = [  0.0,      1.57,  0,  1.57,   1.57,  -1.57]    
        
"""


YASKAWA_GP7_Str = Robot_Parameters_Str(Name='YASKAWA_GP7', Id=1)
# Homogeneous transformation matrix of the base.
#   1\ None: Identity Matrix
#       [[1.0, 0.0, 0.0, 0.0],
#        [0.0, 1.0, 0.0, 0.0],
#        [0.0, 0.0, 1.0, 0.0],
#        [0.0, 0.0, 0.0, 1.0]]
YASKAWA_GP7_Str.T.Base = HTM_Cls(None, np.float64)
# End-effector (tool):
#   1\ None: Identity Matrix
#       [[1.0, 0.0, 0.0, 0.0],
#        [0.0, 1.0, 0.0, 0.0],
#        [0.0, 0.0, 1.0, 0.0],
#        [0.0, 0.0, 0.0, 1.0]]
YASKAWA_GP7_Str.T.End_Effector = HTM_Cls(None, np.float64)
# Denavit-Hartenberg (DH)
YASKAWA_GP7_Str.DH.Standard = np.array([[0.0,                    0.040,    0.330,  -1.5707963267948966],
                                        [-1.5707963267948966,    0.445,    0.0,     0.0],
                                        [0.0,                    0.040,    0.0,    -1.5707963267948966],
                                        [0.0,                    0.0,      0.440,   1.5707963267948966],
                                        [0.0,                    0.0,      0.0,    -1.5707963267948966],
                                        [0.0,                    0.0,      0.080,   0.0]], dtype = np.float64)

YASKAWA_GP7_Str.DH.Modified = np.array([[0.0,                      0.0,        0.330,     0.0],
                                        [-1.5707963267948966,      0.040,      0.0,      -1.5707963267948966],
                                        [ 0.0,                     0.445,      0.0,       0.0],
                                        [ 0.0,                     0.040,      0.440,    -1.5707963267948966],
                                        [ 0.0,                     0.0,        0.0,       1.5707963267948966],
                                        [ 0.32174,                 0.0,        0.080,    -1.5707963267948966]], dtype = np.float64)



# Zero/Home absolute position of each joint.
YASKAWA_GP7_Str.Theta.Zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype = np.float64)
YASKAWA_GP7_Str.Theta.Home = Mathematics.Degree_To_Radian(np.array([0.0, 0.0, 0.0, 0.0, -90.0, 0.0],
                                                                   dtype=np.float64))
# Limits of absolute joint position.
YASKAWA_GP7_Str.Theta.Limit = np.array([[-2.96705972839036, 2.96705972839036],
                                        [-1.13446401379631, 2.53072741539178],
                                        [-2.02458193231342, 4.45058959258554],
                                        [ -3.31612557878923,  3.31612557878923],
                                        [-2.35619449019234, 2.35619449019234],
                                        [ -3.141592653589793,  3.141592653589793]], dtype = np.float64)



# Other parameters of the robot structure.
YASKAWA_GP7_Str.Theta.Name = [f'Joint_1_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}',
                              f'Joint_2_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}',
                              f'Joint_3_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}',
                              f'Joint_4_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}',
                              f'Joint_5_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}',
                              f'Joint_6_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}']
YASKAWA_GP7_Str.Theta.Type = ['R', 'R', 'R', 'R', 'R', 'R']
YASKAWA_GP7_Str.Theta.Axis = ['Z', 'Z', 'Z', 'Z', 'Z', 'Z']
YASKAWA_GP7_Str.Theta.Direction = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.int8)
YASKAWA_GP7_Str.External_Axis = False
#Colliders of the robot structure.
#  1\ Internal.
YASKAWA_GP7_Str.Collider.Base = {f'Base_Collider_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}': OBB_Cls(Box_Cls([0.06060, 0.00000, -0.08317],
                                                                                                                 [0.30120, 0.18052, 0.16633]))}
YASKAWA_GP7_Str.Collider.Theta = {f'Joint_1_Collider_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}': OBB_Cls(Box_Cls([0.00000, 0.00000, 0],
                                                                                                                     [0., 0.0, 0.])),
                              f'Joint_2_Collider_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}': OBB_Cls(Box_Cls([-0., 0.00000, 0.000],
                                                                                                                     [0., 0., 0.])),
                              f'Joint_3_Collider_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}': OBB_Cls(Box_Cls([-0.0, -0.0, -0.00],
                                                                                                                     [0, 0, 0])),
                              f'Joint_4_Collider_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}': OBB_Cls(Box_Cls([-0.00, -0.000, 0.0],
                                                                                                                     [0, 0, 0])),
                              f'Joint_5_Collider_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}': OBB_Cls(Box_Cls([0.00000, -0.0000, 0.00000],
                                                                                                                     [0.0, 0, 0])),
                              f'Joint_6_Collider_{YASKAWA_GP7_Str.Name}_ID_{YASKAWA_GP7_Str.Id:03}': OBB_Cls(Box_Cls([0.00000, 0.00000, 0.0],
                                                                                                                     [0.0, 0.0, 0.0]))}
YASKAWA_GP7_Str.Collider.Offset = 2
#   2\ External.
YASKAWA_GP7_Str.Collider.External = {}
#   Collision pairs.
YASKAWA_GP7_Str.Collider.Pairs = np.array([[0, 4], [0, 5], [0, 6], [0, 3],
                                           [1, 4], [1, 5], [1, 6]], dtype=np.int8)
