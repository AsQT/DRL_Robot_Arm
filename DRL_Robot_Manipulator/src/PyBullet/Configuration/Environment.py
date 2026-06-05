"""
PyBullet Environment Configuration
==================================
Dataclass definitions for robot-environment parameters: search/target configuration
spaces, collision objects, and per-robot environment structures.

Each environment (Default, Collision-Free) is represented by an ``Environment_Str``
instance defining the cuboid bounds of the search space and target space, plus an
optional collision object definition.
"""
# ## =========================================================================== ##
# MIT License
# Copyright (c) 2023 Roman Parak
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ## =========================================================================== ##
# Author   : Roman Parak
# Email    : Roman.Parak@outlook.com
# Github   : https://github.com/rparak
# File Name: ../PyBullet/Configuration/Environment.py
# ## =========================================================================== ##

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Lib.: Robotics Library for Everyone (RoLE)
#   ../RoLE/Transformation/Core
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls


@dataclass
class Cuboid_Str:
    """
    The auxiliary structure of the main parameters of the cuboid.

    Note:
        Private structure.
    """

    # Homogeneous transformation matrix of the cuboid.
    #   Unit [Matrix<float>]
    T: tp.List[tp.List[float]] = field(default_factory=list)
    # The size of the cuboid.
    #   Unit [Vector<float> 1x3]
    Size: tp.List[float] = field(default_factory=list)
    # The color of the cuboid.
    #   Note:
    #       Format: rgba(red, green, blue
    #   Unit [Vector<float> 1x3]
    Color: tp.List[float] = field(default_factory=list)


@dataclass
class Collision_Object_Str:
    """
    The auxiliary structure of the main parameters of the collision object.

    Collision_Object represents the **learning obstacle**: it is loaded into the RL
    environment's collision detector and included in the observation space.  Exactly
    one collision object is supported per environment (e.g. ``cube100`` in
    Collision-Free mode).
    """

    # Homogeneous transformation matrix of the cuboid.
    #   Unit [Matrix<float>]
    T: tp.List[tp.List[float]] = field(default_factory=list)
    # The scale factor of the object.
    #   Unit [float]
    Scale: float = 0.0
    # The color of the object. Format: rgba(red, green, blue, alpha).
    #   Unit [Vector<float> 1x4]
    Color: tp.List[float] = field(default_factory=list)
    # Type of collision object.
    #   Note:
    #       Type = 'Cube' or 'Sphere'
    #   Unit [string]
    Type: str = ''


@dataclass
class Scene_Object_Str:
    """
    The auxiliary structure of the main parameters of a scene-level object.

    Scene objects are static, visual-only scenery elements (e.g. a work table) that
    are always loaded into the PyBullet world but are **never** used as the learning
    obstacle and are **not** included in the observation space or collision checks.

    Unlike :class:`Collision_Object_Str`, multiple scene objects can be defined per
    environment.
    """

    # Human-readable name used as the PyBullet body key.
    #   Unit [string]
    Name: str = ''
    # Project-root-relative path to the object's URDF file.
    #   Unit [string]
    URDF_Relative_Path: str = ''
    # Homogeneous transformation matrix of the object.
    #   Unit [Matrix<float>]
    T: tp.List[tp.List[float]] = field(default_factory=list)
    # The scale factor of the object.
    #   Unit [float]
    Scale: float = 1.0
    # The color of the object. Format: rgba(red, green, blue, alpha).
    #   Unit [Vector<float> 1x4]
    Color: tp.List[float] = field(default_factory=list)
    # Whether the object participates in PyBullet collision detection.
    #   Note:
    #       Set False for visual-only scene elements.
    #   Unit [bool]
    Enable_Collision: bool = False


@dataclass
class Configuration_Space_Str:
    """
    The auxiliary structure of the main parameters of the configuration space.
    """

    # The search (configuration) space indicates the place
    # where the robot can move freely.
    #   Unit [Cuboid_Str(object)]
    Search: Cuboid_Str = field(default_factory=Cuboid_Str)
    # The target (configuration) space indicates the place
    # where the robot aims to reach.
    #   Unit [Cuboid_Str(object)]
    Target: Cuboid_Str = field(default_factory=Cuboid_Str)


@dataclass
class Environment_Str:
    """
    The structure of the main parameters of the environment.

    Attributes:
        Name: Name of the robotic structure.
        C: Configuration space (search and target cuboids).
        Collision_Object: Optional collision object definition (learning obstacle).
        Scene_Objects: List of scene-level objects (visual-only scenery).

    Note:
        - ``Collision_Object`` is the learning obstacle; included in observations and
          used for collision detection.
        - ``Scene_Objects`` are static scene elements (e.g. work table); not used
          by the RL agent and not included in observations.

    Example::

        env = Environment_Str(Name='YASKAWA_GP7')
        env.C.Search = Cuboid_Str(...)
        env.C.Target = Cuboid_Str(...)
        env.Collision_Object = Collision_Object_Str(...)
        env.Scene_Objects = [Scene_Object_Str(...)]
    """

    # The name of the robotic structure for which
    # the configuration space will be defined.
    #   Unit [string]
    Name: str = ''
    # The main parameters of the configuration space.
    #   Unit [Configuration_Space_Str(object)]
    C: Configuration_Space_Str = field(default_factory=Configuration_Space_Str)
    # The main parameters of the collision object.
    #   Unit [None or Collision_Object_Str(object)]
    Collision_Object: Collision_Object_Str = field(default_factory=Collision_Object_Str)
    # Scene-level objects (visual-only static scenery).
    #   Unit [list[Scene_Object_Str]]
    Scene_Objects: tp.List[Scene_Object_Str] = field(default_factory=list)


# ============================================================================
# Environment definitions for YASKAWA_GP7
#
#   Env_ID 0: Default mode — no collision object
#   Env_ID 1: Collision-Free mode — cube100 collision object present
# ============================================================================

# YASKAWA_GP7 — Env_ID 0 (Default)
YASKAWA_GP7_Env_ID_0_Str = Environment_Str(Name='YASKAWA_GP7')
YASKAWA_GP7_Env_ID_0_Str.C.Search = Cuboid_Str(
    HTM_Cls(None, np.float64).Translation(np.array([0.150, -0.350, 0.500], dtype=np.float64)),
    np.array([0.700, 0.700, 0.300], dtype=np.float64), [1.0, 0.984, 0.0])
YASKAWA_GP7_Env_ID_0_Str.C.Target = Cuboid_Str(
    HTM_Cls(None, np.float64).Translation(np.array([0.030, -0.505, 0.410], dtype=np.float64)),
    np.array([0.330, 0.330, 0.100], dtype=np.float64), [0.0, 1.0, 0.0])

# YASKAWA_GP7_Env_ID_0_Str.C.Search = Cuboid_Str(
#     HTM_Cls(None, np.float64).Translation(np.array([0.150, -0.350, 0.500], dtype=np.float64)),
#     np.array([1.700, 1.700, 0.300], dtype=np.float64), [1.0, 0.984, 0.0])
# YASKAWA_GP7_Env_ID_0_Str.C.Target = Cuboid_Str(
#     HTM_Cls(None, np.float64).Translation(np.array([0.550, -0.0, 0.410], dtype=np.float64)),
#     np.array([0.330, 0.330, 0.100], dtype=np.float64), [0.0, 1.0, 0.0])

YASKAWA_GP7_Env_ID_0_Str.Collision_Object = None
YASKAWA_GP7_Env_ID_0_Str.Scene_Objects = [
    Scene_Object_Str(
        Name='Table',
        URDF_Relative_Path='URDFs/Primitives/Table/Table.urdf',
        T=HTM_Cls(None, np.float64).Translation(np.array([0.030, -0.550, 0.18], dtype=np.float64)),
        Scale=1.0,
        Color=[0.5, 0.5, 0.5, 1.0],
        Enable_Collision=False,
    )
]

# YASKAWA_GP7 — Env_ID 1 (Collision-Free)
YASKAWA_GP7_Env_ID_1_Str = Environment_Str(Name='YASKAWA_GP7')
YASKAWA_GP7_Env_ID_1_Str.C.Search = Cuboid_Str(
    HTM_Cls(None, np.float64).Translation(np.array([0.150, -0.350, 0.500], dtype=np.float64)),
    np.array([0.700, 0.700, 0.300], dtype=np.float64), [1.0, 0.984, 0.0])
YASKAWA_GP7_Env_ID_1_Str.C.Target = Cuboid_Str(
    HTM_Cls(None, np.float64).Translation(np.array([0.030, -0.550, 0.470], dtype=np.float64)),
    np.array([0.330, 0.330, 0.010], dtype=np.float64), [0.0, 1.0, 0.0])
YASKAWA_GP7_Env_ID_1_Str.Collision_Object.T = HTM_Cls(None, np.float64).Translation([0.130, -0.500, 0.410])
YASKAWA_GP7_Env_ID_1_Str.Collision_Object.Scale = 1.0
YASKAWA_GP7_Env_ID_1_Str.Collision_Object.Color = [0.85, 0.60, 0.60, 0.75]
YASKAWA_GP7_Env_ID_1_Str.Collision_Object.Type = 'cube100'
YASKAWA_GP7_Env_ID_1_Str.Scene_Objects = [
    Scene_Object_Str(
        Name='Table',
        URDF_Relative_Path='URDFs/Primitives/Table/Table.urdf',
        T=HTM_Cls(None, np.float64).Translation(np.array([0.030, -0.550, 0.180], dtype=np.float64)),
        Scale=1.0,
        Color=[0.5, 0.5, 0.5, 1.0],
        Enable_Collision=False,
    )
]
