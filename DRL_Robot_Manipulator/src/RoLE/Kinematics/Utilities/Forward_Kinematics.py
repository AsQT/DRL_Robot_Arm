
"""
RoLE Kinematics Utilities — Optimised Forward Kinematics
================================================
Pre-computed forward kinematics for the YASKAWA GP7 robot.
Avoids repeated trigonometric computations by caching sin/cos values.

Used by the PyBullet environment for fast TCP pose queries.
"""

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.: Robotics Library for Everyone (RoLE)
#   ../RoLE/Parameters/Robot
import RoLE.Parameters.Robot as Parameters

def __FKF_YASKAWA_GP7(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot_Parameters_Str) -> tp.List[tp.List[float]]:
    """
    Description:
        Calculation of forward kinematics using a fast method for the YASKAWA GP7 robotic arm.

    Args:
        (1) theta [Vector<float>]: Desired absolute joint position in radians / meters.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Matrix<float> 4x4]: Homogeneous end-effector transformation matrix.
    """

    """
    Description:
        Abbreviations for individual functions. Used to speed up calculations.
    """
    t1, t2, t3, t4, t5, t6 = theta
    s1 = np.sin(t1); s2 = np.sin(t2); s3 = np.sin(t3); s4 = np.sin(t4); s5 = np.sin(t5); s6 = np.sin(t6)
    c1 = np.cos(t1); c2 = np.cos(t2); c3 = np.cos(t3); c4 = np.cos(t4); c5 = np.cos(t5); c6 = np.cos(t6)
    s23 = np.sin(t2+t3); c23 = np.cos(t2+t3)
    d6 = 0.080
    # Computation of the homogeneous end-effector transformation matrix {T}
    T = np.array(np.identity(4), dtype=np.float64)
    T[0,0] = s6 * (c4 * s1 - c23 * c1 * s4) + c6 * (c5 * (s1 * s4 + c1 * c23 * c4) + s23 * c1 * s5)
    T[0,1] = c6 * (c4 * s1 - c23 * c1 * s4) - s6 * (c5 * (s1 * s4 + c1 * c23 * c4) + s23 * c1 *s5)
    T[0,2] = s23 * c1 * c5 - s5 * (s1 * s4 + c1 * c23 * c4)
    T[0,3] = 5 * c1 * (0.068 * s23 + 0.077 * c2 + 0.008) - d6 * (s5 * (s1 * s4 + c1 * c23 * c4) - s23 * c1 * c5)
    T[1,0] = -s6 * (c1 * c4 + c23 * s1 * s4) - c6 * (c5 * (c1 * s4 - c4 * s1 * c23) - s23 * s1 * s5)
    T[1,1] = s6 * (c5 * (c1 * s4 - c4 * s1 * c23) - s23 * s1 * s5) - c6 * (c1 * c4 + c23 * s1 * s4)
    T[1,2] = s5 * (c1 * s4 - c4 * s1 * c23) + s23 * s1 * c5
    T[1,3] = 5 * s1 * (0.068 * s23 + 0.077 * c2 + 0.008) + d6 * (s5 * (c1 * s4 - c4 * s1 * c23) + s23 * s1 * c5)
    T[2,0] = -c6 * (c23 * s5 - s23 * c4 * c5) - s23 * s4 * s6
    T[2,1] = s6 * (c23 * s5 - s23 * c4 * c5)-s23 * c6 *s4
    T[2,2] = -c23 * c5 - s23 * c4 * s5
    T[2,3] = 0.385 * s2 - 0.340 * c23 - d6 * (c23 * c5 + s23 * c4 * s5) + 0.330
    T[3,0] = 0.0
    T[3,1] = 0.0
    T[3,2] = 0.0
    T[3,3] = 1.0

    # T_Base @ T_n @ T_EE
    return Robot_Parameters_Str.T.Base @ T @ Robot_Parameters_Str.T.End_Effector
