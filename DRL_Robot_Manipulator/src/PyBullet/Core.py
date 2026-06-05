"""
PyBullet Robot Control Layer
============================
Wraps the PyBullet physics client and exposes a high-level robot interface.

This module provides the :class:`Robot_Cls` class which:
    - Connects to the PyBullet physics server (GUI or DIRECT mode).
    - Loads and initialises the Yaskawa GP7 URDF model.
    - Manages joint control, TCP pose reading, and inverse kinematics.
    - Loads environment scenery (plane, table, and other scene objects).
    - Loads external objects (collision objects, visualisation aids, targets).

The module uses a global gravity constant of 9.81 m/s² and a fixed time step
derived from the ``fps`` property passed at robot construction time.

Object taxonomy:
    - **Collision_Object** (from :mod:`PyBullet.Configuration.Environment`):
      the RL learning obstacle (e.g. ``cube100``).  Loaded automatically in
      ``__init__`` when ``Env_ID`` is set, included in collision checks.
    - **Scene_Object** (from :mod:`PyBullet.Configuration.Environment`):
      visual-only static scenery (e.g. work table).  Loaded automatically in
      ``__init__`` from the ``Scene_Objects`` list.  Never used for collision
      detection or included in observations.
    - **External object** (added via :meth:`Add_External_Object`):
      any URDF object added at runtime (targets, viewpoints, etc.).

The module uses a global gravity constant of 9.81 m/s² and a fixed time step
derived from the ``fps`` property passed at robot construction time.
"""
# Typing (Support for type hints)
import typing as tp
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# PyBullet (Real-Time Physics Simulation) [pip3 install pybullet]
import pybullet as pb
import pybullet_data
# Time (Time access and conversions)
import time
# OS (Operating system interfaces)
import os

# --- Trace toggle: set to True to see detailed PyBullet Core debug output ---
ENABLE_TRACE = False


def _trace(msg: str) -> None:
    """Print `msg` if ENABLE_TRACE is True."""
    if ENABLE_TRACE:
        print(msg)

# Custom Lib.:
#   Robotics Library for Everyone (RoLE)
#       ../RoLE/Parameters/Robot
import RoLE.Parameters.Robot
#       ../RoLE/Trajectory/Utilities
import RoLE.Trajectory.Utilities
#       ../RoLE/Transformation/Core
from RoLE.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
#       ../RoLE/Kinematics/Core
import RoLE.Kinematics.Core as Kinematics
#       ../RoLE/Primitives/Core
from RoLE.Primitives.Core import Box_Cls
#       ../RoLE/Collider/Utilities
from RoLE.Collider.Utilities import Get_Min_Max
#       ../RoLE/Primitives/Core
from RoLE.Collider.Core import AABB_Cls
#   PyBullet
#       ../PyBullet/Utilities
import PyBullet.Utilities
from config_loader import PROJECT_ROOT

# Gravitational constant (m/s²).
CONST_GRAVITY = 9.81
# Locate the project root directory.
CONST_PROJECT_FOLDER = str(PROJECT_ROOT)

class Robot_Cls(object):
    """
    High-level robot interface for a PyBullet physics simulation.

    Connects to the PyBullet client, loads a URDF robot model, manages joint and TCP
    state, and provides inverse kinematics and external object management.

    On construction the class automatically:
        1. Connects to the PyBullet server.
        2. Loads the robot URDF.
        3. Loads scene objects defined in ``Environment_Str.Scene_Objects`` (e.g. table).
        4. Loads the collision object if one is defined for the given ``Env_ID``.
        5. Loads wireframe visualisation aids for the Search and Target configuration
           spaces.

    Args:
        Robot_Parameters_Str: Robot parameter structure from :mod:`RoLE.Parameters.Robot`
            (e.g. ``YASKAWA_GP7_Str``) containing DH parameters, joint limits, home
            configuration, and collision data.
        urdf_file_path: Absolute or project-root-relative path to the robot's URDF file.
        properties: Dict of PyBullet environment properties:

            - ``Enable_GUI`` (bool): Connect in GUI mode (``True``) or DIRECT mode.
            - ``fps`` (int): Simulation frames per second; derives the fixed time step.
            - ``External_Base`` (str or None): Path to an external base URDF, or None.
            - ``Env_ID`` (int): Environment identifier (0 = Default, 1 = Collision-Free).
              Controls which collision objects and scene objects are loaded.
            - ``Camera`` (dict): Visualizer camera parameters (``Yaw``, ``Pitch``,
              ``Distance``, ``Position``).

    Attributes:
        Theta_0: Joint positions at the Zero configuration (read-only property).
        Theta: Current joint positions (read-only property).
        Theta_v: Current joint velocities (read-only property).
        T_EE: 4x4 homogeneous transformation matrix of the end-effector / TCP.
        T_EE_v: 6-D TCP velocity (linear + angular, read-only).

    Example::

        props = {
            'Enable_GUI': True, 'fps': 1000, 'External_Base': None, 'Env_ID': 0,
            'Camera': {'Yaw': 70, 'Pitch': -32, 'Distance': 1.3,
                       'Position': [0.05, -0.10, 0.06]},
        }
        robot = Robot_Cls(YASKAWA_GP7_Str,
                          'URDFs/Robots/YASKAWA_GP7/YASKAWA_GP7.urdf', props)
        robot.Reset('Home')
        print(robot.T_EE)   # 4x4 TCP transformation matrix
        print(robot.Theta)  # current joint positions (radians)
    """

    def __init__(self, Robot_Parameters_Str: RoLE.Parameters.Robot.Robot_Parameters_Str, urdf_file_path: str, properties: tp.Dict) -> None:
        # << PRIVATE >> #
        self.__Robot_Parameters_Str = Robot_Parameters_Str
        self.__external_object = {}
        # Time step.
        self.__delta_time = 1.0/np.float64(properties['fps'])

        # Initialization of the class to generate trajectory.
        self.__Trapezoidal_Cls = RoLE.Trajectory.Utilities.Trapezoidal_Profile_Cls(delta_time=self.__delta_time)

        # Set the parameters of the PyBullet environment.
        self.__Set_Env_Parameters(properties['Enable_GUI'], properties['Camera'])

        # Get the translational and rotational part from the transformation matrix.
        p = self.__Robot_Parameters_Str.T.Base.p.all(); q = self.__Robot_Parameters_Str.T.Base.Get_Rotation('QUATERNION')

        if properties['External_Base'] != None:
            # Load a physics model of the robotic structure base.
            base_id = pb.loadURDF(properties['External_Base'], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 
                                 useFixedBase=True)
            
            # Disable all collisions of the object.
            pb.setCollisionFilterGroupMask(base_id, -1, 0, 0)

            # Load a physics model of the robotic structure.
            self.__robot_id = pb.loadURDF(urdf_file_path, p, [q.x, q.y, q.z, q.w], useFixedBase=True, 
                                          flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            
            # Enable collision detection between specific pairs of links.
            pb.setCollisionFilterPair(self.__robot_id, base_id, -1,-1, 1)
        else:
            # Load a physics model of the robotic structure.
            self.__robot_id = pb.loadURDF(urdf_file_path, p, [q.x, q.y, q.z, q.w], useFixedBase=True,
                                          flags=pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

        # Change robot color to blue (#0000cc).
        num_joints = pb.getNumJoints(self.__robot_id)
        for i in range(-1, num_joints):
            pb.changeVisualShape(
                self.__robot_id,
                i,
                rgbaColor=[0.0, 0.0, 0.8, 1.0],
                physicsClientId=0
            )

        # Load an auxiliary model of the robotic structure, which is represented as a 'ghost'.
        self.__robot_id_ghost = pb.loadURDF(urdf_file_path, p, [q.x, q.y, q.z, q.w], useMaximalCoordinates=False,
                                            useFixedBase=True)
        #   Disable collision of the robot base.
        pb.setCollisionFilterGroupMask(self.__robot_id_ghost, -1, 0, 0)
        #   Disable dynamic parameters of the robot base.
        pb.changeDynamics(self.__robot_id_ghost, linkIndex=-1, linearDamping=0, angularDamping=0, jointDamping=0, mass=0)

        # Obtain the indices of the movable parts of the robotic structure.
        self.__theta_index = []
        for i in range(pb.getNumJoints(self.__robot_id)):
            info = pb.getJointInfo(self.__robot_id , i)
            if info[2] in [pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC]:
                self.__theta_index.append(i)

            # Set the properties of the auxiliary robot structure.
            #   Disable all collisions of the object.
            pb.setCollisionFilterGroupMask(self.__robot_id_ghost, i, 0, 0)
            #   Disable dynamic parameters of the object.
            pb.changeDynamics(self.__robot_id_ghost, linkIndex=i, linearDamping=0, angularDamping=0, jointDamping=0, mass=0)

        # --- TCP link resolution: resolve "ee_link" by name, never hardcode index ---
        self.__tcp_link_name = "ee_link"
        self.__tcp_link_index = self.__Find_Link_Index_By_Name(self.__robot_id, self.__tcp_link_name)
        self.__ghost_tcp_link_index = self.__Find_Link_Index_By_Name(self.__robot_id_ghost, self.__tcp_link_name)
        self.__tool_link_index = self.__Find_Link_Index_By_Name(self.__robot_id, "link_EE")
        self.__link_6_index = self.__Find_Link_Index_By_Name(self.__robot_id, "link_6")
        self.__ghost_tool_link_index = self.__Find_Link_Index_By_Name(self.__robot_id_ghost, "link_EE")
        self.__ghost_link_6_index = self.__Find_Link_Index_By_Name(self.__robot_id_ghost, "link_6")

        print("[INFO] TCP link resolved: " + self.__tcp_link_name + " -> index " + str(self.__tcp_link_index))
        print("[INFO] Ghost TCP link resolved: " + self.__tcp_link_name + " -> index " + str(self.__ghost_tcp_link_index))
        print("[INFO] Tool link (link_EE) index: " + str(self.__tool_link_index))
        print("[INFO] link_6 index: " + str(self.__link_6_index))
        print("[INFO] Ghost tool link (link_EE) index: " + str(self.__ghost_tool_link_index))
        print("[INFO] Ghost link_6 index: " + str(self.__ghost_link_6_index))
        # Verify fixed offset: ee_link should be 0.090 m from link_EE along local Z.
        if self.__tool_link_index is not None and self.__tcp_link_index is not None:
            tool_pose = self.__Get_Link_Pose(self.__robot_id, self.__tool_link_index)
            tcp_pose = self.__Get_Link_Pose(self.__robot_id, self.__tcp_link_index)
            if tool_pose is not None and tcp_pose is not None:
                dist = float(np.linalg.norm(np.array(tcp_pose[0]) - np.array(tool_pose[0])))
                print("[INFO] |ee_link - link_EE| = " + f"{dist:.6f}" + " m  (expected: 0.090000 m)")

        # --- Make the final visible tool mesh (link_EE) black on the main robot ---
        # ee_link is the TCP frame with no mesh; link_EE carries the actual visible mesh.
        if self.__tool_link_index is not None:
            pb.changeVisualShape(
                self.__robot_id,
                self.__tool_link_index,
                rgbaColor=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=0
            )
            print("[INFO] Main robot final visible link link_EE set to black.")

        # --- Ghost visual shape setup: make all links invisible initially, then __Reset_Ghost_Structure controls visibility ---
        self.__Set_Ghost_Visibility(False, [0.0, 0.75, 0.0])

        # Obtain the structure of the main parameters of the environment for the defined robotic arm.
        self.__Env_Structure = PyBullet.Utilities.Get_Environment_Structure(self.__Robot_Parameters_Str.Name, properties['Env_ID'])
        #   Add the cube of the search (configuration) space and get the vertices of the defined cube.[0.60, 1.0, 0.60]
        self.__vertices_C_search = PyBullet.Utilities.Add_Wireframe_Cuboid(self.__Env_Structure.C.Search.T, self.__Env_Structure.C.Search.Size, 
                                                                           self.__Env_Structure.C.Search.Color, 2.5)
        #   Add the cube of the target (configuration) space and get the vertices of the defined cube.
        self.__vertices_C_target = PyBullet.Utilities.Add_Wireframe_Cuboid(self.__Env_Structure.C.Target.T, self.__Env_Structure.C.Target.Size, 
                                                                           self.__Env_Structure.C.Target.Color, 2.5)

        # Get the home absolute joint positions of a specific environment for a defined robotic arm.
        Robot_Parameters_Str.Theta.Home = PyBullet.Utilities.Get_Robot_Structure_Theta_Home(self.__Robot_Parameters_Str.Name, properties['Env_ID'])

        # --- Compute Home TCP pose using PyBullet FK on ee_link via the ghost robot ---
        # This replaces the old RoLE Forward_Kinematics call with PyBullet FK,
        # ensuring the Home orientation reflects the true URDF TCP frame (ee_link).
        home_tcp_pos, home_quat_xyzw, home_quat_wxyz = self.__Get_TCP_Pose_For_Joints(
            self.__robot_id_ghost,
            self.__ghost_tcp_link_index,
            self.__Robot_Parameters_Str.Theta.Home,
        )
        self.__q_Home = home_quat_wxyz
        self.__p_Home_TCP = home_tcp_pos
        print(f"[INFO] Home TCP pose from PyBullet FK (ee_link): p={home_tcp_pos.tolist()}, q_wxyz={home_quat_wxyz.tolist()}")
        # Load the collision object if one is defined for this environment.
        # Collision_Object is the RL learning obstacle — it is added to the robot's
        # external collider dictionary and included in the observation space.
        if self.__Env_Structure.Collision_Object is not None:
            object_id = self.Add_External_Object(
                f'{CONST_PROJECT_FOLDER}/URDFs/Primitives/{self.__Env_Structure.Collision_Object.Type}/{self.__Env_Structure.Collision_Object.Type}.urdf',
                f'{self.__Env_Structure.Collision_Object.Type}_Collision',
                self.__Env_Structure.Collision_Object.T,
                self.__Env_Structure.Collision_Object.Color,
                self.__Env_Structure.Collision_Object.Scale,
                True
            )
            # Get the real AABB size of the object to draw wireframe with correct dimensions.
            (min_AABB, max_AABB) = pb.getAABB(object_id)
            aabb_size = [max_AABB[0] - min_AABB[0],
                         max_AABB[1] - min_AABB[1],
                         max_AABB[2] - min_AABB[2]]
            _ = PyBullet.Utilities.Add_Wireframe_Cuboid(
                self.__Env_Structure.Collision_Object.T,
                aabb_size,
                self.__Env_Structure.Collision_Object.Color[0:3],
                2.5
            )

        # Load scene objects defined in Environment_Str.Scene_Objects.
        # Scene objects are static, visual-only scenery (e.g. work table).  They are
        # NOT the learning obstacle, are NOT included in observations, and are NOT
        # used for collision detection.  Multiple scene objects are supported.
        for scene_obj in self.__Env_Structure.Scene_Objects:
            scene_obj_id = self.Add_External_Object(
                f'{CONST_PROJECT_FOLDER}/{scene_obj.URDF_Relative_Path}',
                scene_obj.Name,
                scene_obj.T,
                scene_obj.Color,
                scene_obj.Scale,
                scene_obj.Enable_Collision,
            )
            # Convenience: store a direct reference to the table body for potential
            # future use, but only when collision is disabled (table is visual-only).
            if scene_obj.Name == 'Table' and not scene_obj.Enable_Collision:
                self.__table_body_id = scene_obj_id

    def Add_Environment(self) -> None:
        """
        .. deprecated::
            Table loading is now handled automatically by ``Robot_Cls.__init__()``
            through the ``Scene_Objects`` field of :class:`Environment_Str`.
            Calling this method after ``__init__`` will **not** create a duplicate
            table because this method is now idempotent.

        Description:
            Load the visual-only table into the PyBullet world.

            **DEPRECATED.**  This method is retained for backward compatibility only.
            The table is loaded automatically from ``Environment_Str.Scene_Objects``
            during ``Robot_Cls.__init__()``.  If called again (e.g. by legacy
            scripts) this method is idempotent — it checks whether the table is
            already present and skips loading if so.
        """
        # Idempotency guard: skip if the table was already loaded via Scene_Object_Str
        # in __init__(), which stores it under the key "Table" in __external_object.
        if 'Table' in self.__external_object:
            print("[INFO] Table already loaded via Scene_Object_Str; skipping Add_Environment().")
            return

        self.__table_body_id = pb.loadURDF(
            f'{CONST_PROJECT_FOLDER}/URDFs/Primitives/Table/Table.urdf',
            basePosition=[0.030, -0.475, 0.0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
        )
        pb.setCollisionFilterGroupMask(self.__table_body_id, -1, 0, 0)
        self.__external_object['Table'] = self.__table_body_id

        num_joints = pb.getNumJoints(self.__table_body_id, physicsClientId=0)
        for i in range(-1, num_joints):
            pb.changeVisualShape(
                self.__table_body_id,
                i,
                rgbaColor=[0.5, 0.5, 0.5, 1.0],
                physicsClientId=0
            )

        print("[INFO] Table loaded (visual only) at [0.030, -0.475, 0.0]")

    def __Set_Env_Parameters(self, enable_gui: bool, camera_parameters: tp.Dict) -> None:
        """
        Description:
            A function to set the parameters of the PyBullet environment.

        Args:
            (1) enable_gui [bool]: Enable/disable the PyBullet GUI.
            (2) camera_parameters [Dictionary {'Yaw': float, 'Pitch': float, 'Distance': float, 
                                               'Position': Vector<float> 1x3}]: The parameters of the camera.
                                                                                    Note:
                                                                                        'Yaw': Yaw angle of the camera.
                                                                                        'Pitch': Pitch angle of the camera.
                                                                                        'Distance': Distance between the camera 
                                                                                                    and the camera target.
                                                                                        'Position': Camera position in Cartesian 
                                                                                                    world space coordinates. 
        """

        # Connect to the physics simulation and create an environment with additional properties.
        if enable_gui == True:
            pb.connect(pb.GUI, options='--background_color_red=0.0 --background_color_green=0.0 --background_color_blue=0.0')
        else:
            pb.connect(pb.DIRECT)
        # Additional properties.
        pb.setTimeStep(self.__delta_time)
        pb.setRealTimeSimulation(0)
        pb.resetSimulation()
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0.0, 0.0, -CONST_GRAVITY)

        # Set the parameters of the camera.
        pb.resetDebugVisualizerCamera(cameraYaw=camera_parameters['Yaw'], cameraPitch=camera_parameters['Pitch'], cameraDistance=camera_parameters['Distance'], 
                                      cameraTargetPosition=camera_parameters['Position'])
        
        # Configure settings for the built-in OpenGL visualizer.
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)

        # Load a physics model of the plane.
        plane_id = pb.loadURDF(f'{CONST_PROJECT_FOLDER}/URDFs/Primitives/Plane/Plane.urdf', globalScaling=0.20, useMaximalCoordinates=True, useFixedBase=True)
        #   Change the texture of the loaded object.
        pb.changeVisualShape(plane_id, -1, textureUniqueId=pb.loadTexture(f'{CONST_PROJECT_FOLDER}/Textures/Plane.png'))
        pb.changeVisualShape(plane_id, -1, rgbaColor=[0.55, 0.55, 0.55, 0.95])

    @property
    def is_connected(self) -> bool:
        """
        Description:
            Information about a successful connection to the physical server.

        Returns:
            (1) parameter [bool]: The result is 'True' if it is connected, 'False' if it is not.
        """

        return pb.isConnected()

    def __Find_Link_Index_By_Name(self, body_id: int, link_name: str) -> tp.Union[int, None]:
        """
        Find the PyBullet link index for a named child link.

        Args:
            body_id: PyBullet body ID.
            link_name: Exact name of the child link (as declared in URDF <child link=...>).

        Returns:
            PyBullet joint index whose child link matches `link_name`, or None if not found.
        """
        for i in range(pb.getNumJoints(body_id)):
            ji = pb.getJointInfo(body_id, i)
            child_link_name = ji[12].decode(errors="ignore")
            if child_link_name == link_name:
                return i
        return None

    def __Print_Link_Table(self, body_id: int, title: str = "ROBOT") -> None:
        """
        Print a table of all PyBullet joint/link indices for debugging.

        Args:
            body_id: PyBullet body ID.
            title: Label printed in the header.
        """
        print(f"\n=== {title} LINK TABLE ===")
        print("base index = -1, link name = base_link")
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
                f"idx={i:2d} | joint={joint_name:15s} | "
                f"type={type_name:9s} | child_link={child_link_name:15s} | "
                f"parent_idx={parent_index}"
            )

    def __Get_Link_Pose(self, body_id: int, link_index: int) -> tp.Union[tp.Tuple[tp.List[float], tp.List[float]], None]:
        """
        Read the world-frame position and quaternion of a link.

        Args:
            body_id: PyBullet body ID.
            link_index: PyBullet link/joint index.

        Returns:
            Tuple of (position [x,y,z], quaternion [x,y,z,w]), or None on failure.
        """
        ls = pb.getLinkState(body_id, link_index, computeForwardKinematics=True)
        if ls is None:
            return None
        if len(ls) >= 6 and ls[4] is not None and ls[5] is not None:
            return ls[4], ls[5]
        return ls[0], ls[1]

    def __Set_Ghost_Visibility(self, visible: bool, color: tp.List[float]) -> None:
        """
        Set transparency and color for all ghost robot links, including fixed links.

        This method ensures that fixed links (e.g. link_EE and ee_link) are visible
        when the ghost is enabled, fixing the issue where only active joint links
        were being recolored by __Reset_Ghost_Structure.

        Args:
            visible: If True, ghost is shown with alpha=0.30. If False, fully transparent.
            color: RGB color as [r, g, b], each in [0, 1].
        """
        alpha = 0.30 if visible else 0.0
        rgba = [float(color[0]), float(color[1]), float(color[2]), float(alpha)]

        # Base link
        pb.changeVisualShape(self.__robot_id_ghost, linkIndex=-1, rgbaColor=rgba)

        # All links including fixed ones (link_EE, ee_link)
        for link_idx in range(pb.getNumJoints(self.__robot_id_ghost)):
            pb.changeVisualShape(self.__robot_id_ghost, linkIndex=link_idx, rgbaColor=rgba)

        # Make link_EE more visible when ghost is shown — this is the final visible tool mesh.
        if self.__ghost_tool_link_index is not None:
            tool_alpha = 0.45 if visible else 0.0
            tool_rgba = [float(color[0]), float(color[1]), float(color[2]), float(tool_alpha)]
            pb.changeVisualShape(
                self.__robot_id_ghost,
                linkIndex=self.__ghost_tool_link_index,
                rgbaColor=tool_rgba,
            )

        if ENABLE_TRACE:
            print("[TRACE-Core] Ghost visual shapes after __Set_Ghost_Visibility:")
            for vs in pb.getVisualShapeData(self.__robot_id_ghost):
                link_idx = vs[1]
                # Only print link_EE entry to avoid spam
                if self.__ghost_tool_link_index is not None and link_idx == self.__ghost_tool_link_index:
                    print(f"  [TRACE-Core] link_EE visual shape: {vs}")

    def __Get_TCP_Pose_For_Joints(self, body_id: int, tcp_link_index: int,
                                   theta: tp.List[float]) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Temporarily apply joint angles to a body, read TCP pose via PyBullet FK, then restore.

        Args:
            body_id: PyBullet body ID (main robot or ghost).
            tcp_link_index: Resolved TCP link index (typically ee_link).
            theta: Joint angle vector to apply.

        Returns:
            Tuple of (position [x,y,z], quaternion [x,y,z,w], quaternion [w,x,y,z]).
            The quaternion is converted from PyBullet's [x,y,z,w] to [w,x,y,z] convention.
        """
        theta = np.asarray(theta, dtype=np.float64)

        old_theta = []
        for th_index in self.__theta_index:
            old_theta.append(pb.getJointState(body_id, th_index)[0])

        for th_i, th_index in zip(theta, self.__theta_index):
            pb.resetJointState(body_id, th_index, float(th_i))

        link_state = pb.getLinkState(body_id, tcp_link_index, computeForwardKinematics=True)
        if len(link_state) >= 6 and link_state[4] is not None and link_state[5] is not None:
            pos = np.array(link_state[4], dtype=np.float64)
            quat_xyzw = np.array(link_state[5], dtype=np.float64)
        else:
            pos = np.array(link_state[0], dtype=np.float64)
            quat_xyzw = np.array(link_state[1], dtype=np.float64)

        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

        for old_th_i, th_index in zip(old_theta, self.__theta_index):
            pb.resetJointState(body_id, th_index, float(old_th_i))

        return pos, quat_xyzw, quat_wxyz

    @property
    def Theta_0(self) -> tp.List[float]:
        """
        Description:
            Get the zero (home) absolute position of the joint in radians/meter.

        Returns:
            (1) parameter [Vector<float> 1xn]: Zero (home) absolute joint position in radians / meters.
                                                Note:
                                                    Where n is the number of joints.
        """
                
        return self.__Robot_Parameters_Str.Theta.Zero
    
    @property
    def Theta(self) -> tp.List[float]: 
        """
        Description:
            Get the absolute positions of the robot's joints.

        Returns:
            (1) parameter [Vector<float> 1xn]: Current absolute joint position in radians / meters.
                                                Note:
                                                    Where n is the number of joints.
        """
                
        theta_out = np.zeros(self.__Robot_Parameters_Str.Theta.Zero.size, 
                             dtype=np.float64)
        for i, th_index in enumerate(self.__theta_index):
            theta_out[i] = pb.getJointState(self.__robot_id, th_index)[0]

        return theta_out
    
    @property
    def Theta_v(self) -> tp.List[float]:
        """
        Description:
            Get the velocity of the robot's joints.

        Returns:
            (1) parameter [Vector<float> 1xn]: Current velocities in radians / meters per second.
                                                Note:
                                                    Where n is the number of joints.
        """

        theta_v_out = np.zeros(self.__Robot_Parameters_Str.Theta.Zero.size, 
                               dtype=np.float64)
        for i, th_index in enumerate(self.__theta_index):
            theta_v_out[i] = pb.getJointState(self.__robot_id, th_index)[1]

        return theta_v_out

    @property
    def T_EE(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the homogeneous transformation matrix of the robot end-effector (TCP).
            Reads TCP pose directly from PyBullet via getLinkState (link index resolved
            to "ee_link" at __init__ time), NOT from the internal RoLE FK model.

        Returns:
            (1) parameter [Matrix<float> 4x4]: Homogeneous transformation matrix of the End-Effector / TCP.
        """
        link_state = pb.getLinkState(self.__robot_id, self.__tcp_link_index, computeForwardKinematics=True)
        tcp_pos = np.array(link_state[4])
        # PyBullet returns quaternion in [x, y, z, w] order.
        # RoLE HTM_Cls expects [w, x, y, z] — swap accordingly.
        tcp_quat_xyzw = np.array(link_state[5], dtype=np.float64)
        tcp_quat_wxyz = np.array([
            tcp_quat_xyzw[3],
            tcp_quat_xyzw[0],
            tcp_quat_xyzw[1],
            tcp_quat_xyzw[2],
        ], dtype=np.float64)
        return HTM_Cls(None, np.float64).Rotation(tcp_quat_wxyz, 'QUATERNION').Translation(tcp_pos.tolist())

    @property
    def T_EE_v(self):
        """
        Description:
            Get the linear and angular velocity of the robot's end-effector.

        Returns:
            (1) paramter [Vector<float> 1x6]: The linear (Vector<float> 1x3) and angular (Vector<float> 1x3) velocity 
                                              of the robot's end effector.
        """

        # Get the matrix of the geometric Jacobian.
        J = Kinematics.Get_Geometric_Jacobian(self.Theta, self.__Robot_Parameters_Str)
        #   Linear Velocity of the End-Effector.
        J_P = J[0:3, 0::]
        #   Angular Velocity of the End-Effector.
        J_O = J[3::, 0::]

        return np.concatenate(((J_P @ self.Theta_v).flatten(), 
                               (J_O @ self.Theta_v).flatten()))
    
    def Get_Camera_Parameters(self) -> tp.Dict:
        """
        Description:
            Obtain the camera's parameters.

            Note:
                The obtained parameters can be used as one of the input properties of the class.

        Returns:
            (1) parameter [Dictionary {'Yaw': float, 'Pitch': float, 'Distance': float, 
                                       'Position': Vector<float> 1x3}]: The parameters of the camera.
                                                                            Note:
                                                                                'Yaw': Yaw angle of the camera.
                                                                                'Pitch': Pitch angle of the camera.
                                                                                'Distance': Distance between the camera 
                                                                                            and the camera target.
                                                                                'Position': Camera position in Cartesian 
                                                                                            world space coordinates.
        """

        parameters = pb.getDebugVisualizerCamera()

        return {'Yaw': parameters[8], 'Pitch': parameters[9], 'Distance': parameters[10], 
                'Position': parameters[11]}
    
    def Get_Configuration_Space_Vertices(self, C_type: str):
        """
        Description:
            Get the vertices of the selected configuration space.

        Args:
            (1) C_type [string]: Type of the configuration space.
                                    Note:
                                        C_type = 'Search' or 'Target'

        Returns:
            (1) parameter [Vector<float> 8x3]: Vertices of the selected configuration space.
        """

        try:
            assert C_type in ['Search', 'Target']

            if C_type == 'Search':
                return self.__vertices_C_search
            else:
                return self.__vertices_C_target

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print('[ERROR] Incorrect configuration type selected. The selected mode must be chosen from the two options (Search, Target).')

    def Step(self) -> None:
        """
        Description:
            A function to perform all the actions in a single forward dynamics 
            simulation step extended with a time step value.
        """

        pb.stepSimulation()

        # The time to approximate and update the state of the dynamic system.
        time.sleep(self.__delta_time)

    def Disconnect(self) -> None:
        """
        Description:
            A function to disconnect the created environment from a physical server.
        """
                
        if self.is_connected == True:
            pb.disconnect()

    def Add_External_Object(self, urdf_file_path: str, name: str, T: HTM_Cls, color: tp.Union[None, tp.List[float]],
                            scale: float, enable_collision: bool) -> int:
        """
        Description:
            A function to add external objects with the *.urdf extension to the PyBullet environment.

        Args:
            (1) urdf_file_path [string]: The specified path of the object file with the extension '*.urdf'.
            (2) name [string]: The name of the object.
            (3) T [Matrix<float> 4x4]: Homogeneous transformation matrix of the object.
            (4) color [Vector<float> 1x4]: The color of the object.
                                            Note:
                                                Format: rgba(red, green, blue, alpha)
            (5) scale [float]: The scale factor of the object.
            (6) enable_collision [bool]: Information on whether or not the object is to be exposed
                                         to collisions.

        Returns:
            (1) parameter [int]: The PyBullet body ID of the loaded object.
                                 The AABB collider is only registered when enable_collision is True.
        """

        # Get the translational and rotational part from the transformation matrix.
        p = T.p.all(); q = T.Get_Rotation('QUATERNION')

        # --- DIAGNOSTIC: print quaternion PyBullet receives ---
        _trace(f'[TRACE-Core-ADD] pyb_reset quaternion [x,y,z,w] = [{q.x}, {q.y}, {q.z}, {q.w}]')

        # Load a physics model of the object.
        #   Note:
        #       Set the object position to 'Zero'.
        object_id = pb.loadURDF(urdf_file_path, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], globalScaling=scale, useMaximalCoordinates=False, 
                                useFixedBase=True)
        #   Store the object ID and object name into the dictionary.
        self.__external_object[name] = object_id

        # Get the minimum and maximum X, Y, and Z values from the AABB coordinates of the object.
        (min_AABB, max_AABB) = pb.getAABB(object_id)

        # Set the object position to the desired position defined by the function 
        # input parameters.
        pb.resetBasePositionAndOrientation(object_id, p, [q.x, q.y, q.z, q.w])
        # Disable all collisions of the object.
        #   Note:
        #       Collisions will be solved internally.
        pb.setCollisionFilterGroupMask(object_id, -1, 0, 0)

        # Set the properties of the added object.
        #   Color.
        if color is not None:
            pb.changeVisualShape(object_id, linkIndex=-1, rgbaColor=color)
        #   Collision: only register AABB collider when collision is enabled.
        #   The PyBullet body is always returned so visual-only objects can be managed too.
        if enable_collision == True:
            # Add a collider (type AABB) as a part of the robotic arm structure.
            self.__Robot_Parameters_Str.Collider.External[name] = AABB_Cls(Box_Cls([0.0, 0.0, 0.0],
                                                                                  [max_AABB[0] - min_AABB[0],
                                                                                   max_AABB[1] - min_AABB[1],
                                                                                   max_AABB[2] - min_AABB[2]]))
            # Axis-aligned Bounding Boxe (AABB) transformation according to the input homogeneous
            # transformation matrix.
            self.__Robot_Parameters_Str.Collider.External[name].Transformation(T)

        print(f"[INFO] Scene object loaded: {name}")
        return object_id

    def Transformation_External_Object(self, name: str, T: HTM_Cls, enable_collision: bool) -> None:
        """
        Description:
            A function to transform external objects that have been added by the 'Add_External_Object' 
            function into the PyBullet environment.

        Args:
            (1) name [string]: The name of the object.
            (2) T [Matrix<float> 4x4]: Homogeneous transformation matrix of the object.
            (3) enable_collision [bool]: Information on whether or not the object is to be exposed 
                                         to collisions.
        """

        try:
            assert name in self.__external_object.keys()

            # Get the translational and rotational part from the transformation matrix.
            p = T.p.all(); q = T.Get_Rotation('QUATERNION')

            # --- DIAGNOSTIC: print quaternion PyBullet receives ---
            _trace(f'[TRACE-Core-XFORM] pyb_reset quaternion [x,y,z,w] = [{q.x}, {q.y}, {q.z}, {q.w}]')

            # Set the object position to the desired position defined by the function
            # input parameters.
            pb.resetBasePositionAndOrientation(self.__external_object[name], p, [q.x, q.y, q.z, q.w])
            
            # Axis-aligned Bounding Boxe (AABB) transformation according to the input homogeneous 
            # transformation matrix.
            if enable_collision == True:
                self.__Robot_Parameters_Str.Collider.External[name].Transformation(T)

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')

    def Remove_External_Object(self, name: str) -> None:
        """
        Description:
            A function to remove a specific model with the *.urdf extension from the PyBullet environment
            that was added using the 'Add_External_Object' function of the class.

            Note:
                The function also removes external collider added to the robotic structure.

        Args:
            (1) name [string]: The name of the object.
        """

        if name in self.__external_object.keys():
            pb.removeBody(self.__external_object[name])
            del self.__external_object[name]

        if name in self.__Robot_Parameters_Str.Collider.External.keys():
            del self.__Robot_Parameters_Str.Collider.External[name]

    def Remove_All_External_Objects(self) -> None:
        """
        Description:
            A function to remove all models with the *.urdf extension from the PyBullet environment 
            that were added using the 'Add_External_Object' function of the class.

            Note:
                The function also removes external colliders added to the robotic structure.
        """

        for _, external_obj in enumerate(self.__external_object.values()):
            pb.removeBody(external_obj)

        self.__external_object = {}; self.__Robot_Parameters_Str.Collider.External = {}

    def Generate_Random_T_EE(self, C_type: str, visibility: bool) -> tp.List[tp.List[float]]:
        """
        Description:
            A function that generates the homogeneous transformation matrix of a random end-effector
            position within the defined configuration space.

            Note:
                Orientation is fixed: TCP Z-axis always points upward (+Z world).
                Only position (x, y, z) is randomized. No random rotation.

        Args:
            (1) C_type [string]: Type of the configuration space.
                                    Note:
                                        C_type = 'Search' or 'Target'
            (2) visibility [bool]: Information about whether the random point will be displayed
                                   in the PyBullet environment or not.

        Returns:
            (1) parameter [Matrix<float> 4x4]: Homogeneous transformation matrix of a random end-effector position
                                               within the defined configuration space.
        """

        try:
            assert C_type in ['Search', 'Target']

            # --- FIXED ORIENTATION: TCP Z-axis always points downward (-Z world) ---
            # Quaternion format for HTM_Cls: [w, x, y, z].
            # Rotation 180° around X-axis: q = [w=0, x=1, y=0, z=0].
            # This maps world +Z → world -Z (TCP Z points downward).
            # NOTE: numpy array [1,0,0,0] is [w=1,x=0,y=0,z=0] = IDENTITY (no rotation)!
            q_Down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
            # Print once on first call only (static flag).
            if not hasattr(self, '_Generate_Random_T_EE__orientation_printed'):
                self.__orientation_printed = True
                _trace(f'[TRACE-Core] Fixed target orientation (quaternion): {q_Down}')
                _trace(f'[TRACE-Core] Target Z-axis: [0, 0, -1] = downward (-Z world)')

            if C_type == 'Search':
                # Get the minimum and maximum X, Y, Z values of the input vertices.
                (min_vec3, max_vec3) = Get_Min_Max(self.__vertices_C_search)

            else:
                # Get the minimum and maximum X, Y, Z values of the input vertices.
                (min_vec3, max_vec3) = Get_Min_Max(self.__vertices_C_target)

            x = np.random.uniform(min_vec3[0], max_vec3[0])
            y = np.random.uniform(min_vec3[1], max_vec3[1])
            z = np.random.uniform(min_vec3[2], max_vec3[2])

            _trace(f'[TRACE-Core] Random target position: [{x:.4f}, {y:.4f}, {z:.4f}]')

            # Build HTM with fixed downward orientation + random position.
            T = HTM_Cls(None, np.float64).Rotation(q_Down, 'QUATERNION').Translation([x, y, z])

            if visibility == True:
                # Removal of external objects corresponding to a random point.
                self.Remove_External_Object('T_EE_Rand_Viewpoint')

                # --- DIAGNOSTIC: verify logical quaternion from the HTM ---
                q_from_htm = T.Get_Rotation('QUATERNION')
                _trace(f'[TRACE-Core] T_htm quaternion [w,x,y,z] = {q_from_htm.all()}')

                # Adding external objects corresponding to a random point.
                self.Add_External_Object(f'{CONST_PROJECT_FOLDER}/URDFs/Viewpoint/Viewpoint.urdf', 'T_EE_Rand_Viewpoint', T,
                                         None, 0.3, False)
            return T

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print('[ERROR] Incorrect configuration type selected. The selected mode must be chosen from the two options (Search, Target).')

    def __Reset_Ghost_Structure(self, theta: tp.List[float], visibility: bool, color: tp.Union[None, tp.List[float]]) -> None:
        """
        Reset the ghost robot's joint positions and update visual transparency for all links.

        Fixed links (link_EE, ee_link) are included via __Set_Ghost_Visibility,
        which fixes the previous issue where only active joint links were recolored.

        Args:
            (1) theta [Vector<float> 1xn]: Desired absolute joint position in radians / meters.
            (2) visibility [bool]: If True, ghost is shown semi-transparently. If False, invisible.
            (3) color [None or Vector<float> 1x3]: RGB color of the ghost.
        """

        # Reset joint positions for active joints only.
        for th_i, th_index in zip(theta, self.__theta_index):
            pb.resetJointState(self.__robot_id_ghost, th_index, th_i)

        # Apply visibility to ALL links including fixed links (link_EE, ee_link).
        # Color defaults to green if None is passed.
        if color is None:
            color = [0.0, 0.75, 0.0]
        self.__Set_Ghost_Visibility(visibility, color)
    
    def Reset(self, mode: str, theta: tp.Union[None, tp.List[float]] = None, enable_ghost: bool = False) -> bool:
        """
        Description:
            Function to reset the absolute position of the robot joints from the selected mode.

            Note:
                The Zero/Home modes are predefined in the robot structure and the Individual mode is used 
                to set the individual position defined in the function input parameter.

        Args:
            (1) mode [string]: Possible modes to reset the absolute position of the joints.
                                Note:
                                    mode = 'Zero', 'Home' or 'Individual'
            (2) theta [Vector<float> 1xn]: Desired absolute joint position in radians / meters. Used only in individual 
                                           mode.
                                            Note:
                                                Where n is the number of joints.
            (3) enable_ghost [bool]: Enable visibility of the auxiliary robotic structure, which is represented as a 'ghost'.

        Returns:
            (1) parameter [bool]: The result is 'True' if the robot is in the desired position,
                                  and 'False' if it is not.
        """
                
        try:
            assert mode in ['Zero', 'Home', 'Individual']

            if mode == 'Individual':
                assert self.__Robot_Parameters_Str.Theta.Zero.size == theta.size
                
                theta_internal = theta
            else:
                theta_internal = self.Theta_0 if mode == 'Zero' else self.__Robot_Parameters_Str.Theta.Home

            if enable_ghost == True:
                self.__Reset_Ghost_Structure(theta_internal, True, [0.70, 0.85, 0.60])
            else:
                for i, (th_i, th_i_limit, th_index) in enumerate(zip(theta_internal, self.__Robot_Parameters_Str.Theta.Limit, 
                                                                     self.__theta_index)):
                    if th_i_limit[0] <= th_i <= th_i_limit[1]:
                        # Reset the state (position) of the joint.
                        pb.resetJointState(self.__robot_id, th_index, th_i) 
                    else:
                        print(f'[WARNING] The desired input joint {th_i} in index {i} is out of limit.')
                        return False
                
            return True

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            if mode not in ['Zero', 'Home', 'Individual']:
                print('[ERROR] Incorrect reset mode selected. The selected mode must be chosen from the three options (Zero, Home, Individual).')
            if self.__Robot_Parameters_Str.Theta.Zero.size != theta.size:
                print(f'[ERROR] Incorrect number of values in the input variable theta. The input variable "theta" must contain {self.__Robot_Parameters_Str.Theta.Zero.size} values.')

    def Set_Absolute_Joint_Position(self, theta: tp.List[float], properties: tp.Dict = None) -> bool:
        """
        Description:
            Set the absolute position of the robot joints.

            Note:
                To use the velocity control of the robot's joint, it is necessary to change the input 
                parameters of the 'setJointMotorControl2' function from position to:

                    pb.setJointMotorControl2(self.__robot_id, th_index, pb.VELOCITY_CONTROL, 
                                             targetVelocity=th_v_i, force=force),

                and get the velocity from trapezoidal trajectories.

        Args:
            (1) theta [Vector<float> 1xn]: Desired absolute joint position in radians / meters.
                                            Note:
                                                Where n is the number of joints.
            (2) properties [Dictionary {'force': float, 
                                        't_0': float, 't_1': float}] The properties of a function to control the absolute 
                                                                     position of the robot's joints.
                                                                        Note:
                                                                            'force': The maximum motor force used to reach the target value.
                                                                            't_0': Animation start time in seconds.
                                                                            't_1': Animation stop time in seconds.

                                                                            Note 2:
                                                                                If time t_0, t_1 is equal to 'None', the trajectory generation 
                                                                                will be ignored.

        Returns:
            (1) parameter [bool]: The result is 'True' if the robot is in the desired position,
                                  and 'False' if it is not.
        """
                
        try:
            assert self.__Robot_Parameters_Str.Theta.Zero.size == theta.size

            theta = np.array(theta, dtype=np.float64)
            _trace(f'[TRACE-Core-EXEC] Set_Absolute_Joint_Position ENTRY')
            _trace(f'[TRACE-Core-EXEC] theta (commanded) = {theta}')
            _trace(f'[TRACE-Core-EXEC] current Theta = {self.Theta}')

            if None in [properties['t_0'], properties['t_1']]:
                # --- SMOOTH INTERPOLATION PHASE ---
                # Read current theta before starting motion.
                theta_start = self.Theta
                # Number of interpolation substeps — more steps = slower, smoother motion.
                NUM_INTERP_STEPS = 75
                STEP_SLEEP = 0.03  # seconds of real time between substeps

                _trace(f'[TRACE-Core-EXEC] Smooth interpolation: {NUM_INTERP_STEPS} substeps, {STEP_SLEEP}s sleep')
                _trace(f'[TRACE-Core-EXEC] theta_start = {theta_start}')
                _trace(f'[TRACE-Core-EXEC] theta_target = {theta}')

                for interp_step in range(NUM_INTERP_STEPS):
                    # Linear interpolation from current theta to target theta.
                    alpha = (interp_step + 1) / NUM_INTERP_STEPS
                    theta_interp = theta_start + alpha * (theta - theta_start)

                    # Command intermediate joint positions.
                    for i, (th_i, th_i_limit, th_index) in enumerate(zip(theta_interp, self.__Robot_Parameters_Str.Theta.Limit,
                                                                        self.__theta_index)):
                        if th_i_limit[0] <= th_i <= th_i_limit[1]:
                            pb.setJointMotorControl2(self.__robot_id, th_index, pb.POSITION_CONTROL, targetPosition=th_i,
                                                        positionGain=1.0, velocityGain=1.0, force=properties['force'])
                        else:
                            print(f'[WARNING] Interpolated joint {th_i} at index {i} out of limit.')
                            return False

                    # Step simulation and pause for visible motion.
                    self.Step()
                    time.sleep(STEP_SLEEP)

                    _trace(f'[TRACE-Core-EXEC] Interp step {interp_step + 1}/{NUM_INTERP_STEPS}: interp={theta_interp}')

                # --- CONVERGENCE VERIFICATION PHASE ---
                # After interpolation the robot should be near target.
                # Run a final convergence check to confirm it settled.
                CONVERGENCE_TOLERANCE = 0.001  # radians
                MAX_STEPS = 100

                _trace(f'[TRACE-Core-EXEC] Verifying convergence (tolerance={CONVERGENCE_TOLERANCE}, max_steps={MAX_STEPS})')

                for step in range(MAX_STEPS):
                    # Command all joints simultaneously.
                    for i, (th_i, th_i_limit, th_index) in enumerate(zip(theta, self.__Robot_Parameters_Str.Theta.Limit,
                                                                        self.__theta_index)):
                        if th_i_limit[0] <= th_i <= th_i_limit[1]:
                            pb.setJointMotorControl2(self.__robot_id, th_index, pb.POSITION_CONTROL, targetPosition=th_i,
                                                        positionGain=1.0, velocityGain=1.0, force=properties['force'])
                        else:
                            print(f'[WARNING] Desired input joint {th_i} at index {i} is out of limit.')
                            return False

                    # Step simulation.
                    self.Step()
                    time.sleep(STEP_SLEEP)

                    # Read back actual joint positions.
                    theta_actual = self.Theta

                    # Compute max error across all joints.
                    joint_errors = np.abs(theta_actual - theta)
                    max_error = np.max(joint_errors)

                    _trace(f'[TRACE-Core-EXEC] Verify step {step:3d}: commanded={theta}, actual={theta_actual}, max_error={max_error:.6f}')

                    # Check convergence.
                    if max_error <= CONVERGENCE_TOLERANCE:
                        _trace(f'[TRACE-Core-EXEC] CONVERGED at step {step} (max_error={max_error:.6f} <= {CONVERGENCE_TOLERANCE})')
                        break

                if max_error > CONVERGENCE_TOLERANCE:
                    print(f'[WARNING] Convergence not reached after {MAX_STEPS} steps (final max_error={max_error:.6f})')

            else:
                # Generation of multi-axis position trajectories from input parameters.
                theta_arr = []
                for _, (th_actual, th_desired) in enumerate(zip(self.Theta, theta)):
                    (theta_arr_i, _, _) = self.__Trapezoidal_Cls.Generate(th_actual, th_desired, 0.0, 0.0,
                                                                            properties['t_0'], properties['t_1'])
                    theta_arr.append(theta_arr_i)

                for _, theta_arr_i in enumerate(np.array(theta_arr, dtype=np.float64).T):
                    for i, (th_i, th_i_limit, th_index) in enumerate(zip(theta_arr_i, self.__Robot_Parameters_Str.Theta.Limit,
                                                                            self.__theta_index)):
                        if th_i_limit[0] <= th_i <= th_i_limit[1]:
                            # Control of the robot's joint positions.
                            pb.setJointMotorControl2(self.__robot_id, th_index, pb.POSITION_CONTROL, targetPosition=th_i,
                                                        positionGain=1.0, velocityGain=1.0, force=properties['force'])
                        else:
                            print(f'[WARNING] The desired input joint {th_i} in index {i} is out of limit.')
                            return False

                    # Update the state of the dynamic system.
                    self.Step()

            return True
            
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrect number of values in the input variable theta. The input variable "theta" must contain {self.__Robot_Parameters_Str.Theta.Zero.size} values.')

    def Get_Inverse_Kinematics_Solution(self, T: tp.List[tp.List[float]], ik_solver_properties: tp.Dict, enable_ghost: bool) -> tp.Dict[bool, tp.List[float]]:
        """
        Compute inverse kinematics (IK) for the robot using PyBullet.

        After calling the PyBullet IK solver, the method validates the solution by
        applying the returned joint angles to the internal ghost robot and verifying
        that the resulting TCP position is within a configurable tolerance.

        Note:
            Success requires ALL of the following to be true:
                (1) IK returned a finite joint vector.
                (2) Joint count matches the active DOF.
                (3) No joint was clipped to stay within joint limits.
                (4) Ghost-robot TCP position error <= ik_position_tolerance (default 1 cm).

            If any joint is out of limits the solution is REJECTED (not clipped
            and accepted as before).  This makes the success flag trustworthy.

        Args:
            (1) T: Homogeneous transformation matrix of the desired TCP position.
            (2) ik_solver_properties: Dict with IK solver settings:
                    'num_of_iteration' (int): Iterations passed to PyBullet maxNumIterations.
                        Defaults to 1000.
                    'tolerance' (float): Residual threshold passed to PyBullet residualThreshold.
                        Defaults to 1e-6.
                    'ik_position_tolerance' (float): Ghost-TCP position error threshold in
                        metres.  Defaults to 0.01 (1 cm).
                    'use_orientation' (bool): If True, 6-D IK (position + orientation).
                        If False (default), position-only IK.  Position-only is correct for
                        Cartesian reaching where reward/observation/action are position-based.
                    'delta_time' (float): Ignored by the IK solve.  Belongs in motion config.
            (3) enable_ghost (bool): Enable visibility of the ghost robot.

        Returns:
            (1) bool: True only if IK converged, joints are in range, and
                      ghost-robot TCP position error is within tolerance.
            (2) theta: Joint angle solution (radians/metres).
        """

        _trace(f'[TRACE-Core] Get_Inverse_Kinematics_Solution ENTRY, robot_id={self.__robot_id}')
        if isinstance(T, (list, np.ndarray)):
            T = HTM_Cls(T, np.float64)

        target_position = T.p.all()
        target_quat_wxyz = T.Get_Rotation('QUATERNION')  # [w, x, y, z]
        target_orientation = np.array([
            target_quat_wxyz.x, target_quat_wxyz.y, target_quat_wxyz.z, target_quat_wxyz.w
        ])  # [x, y, z, w] for PyBullet
        _trace(f'[TRACE-Core] target_position = {target_position}')
        _trace(f'[TRACE-Core] target_orientation [x,y,z,w] = {target_orientation}')

        # --- Robust defaults from properties dict ---
        max_num_iterations = int(ik_solver_properties.get('num_of_iteration', 1000))
        residual_threshold = float(ik_solver_properties.get('tolerance', 1e-6))
        ik_position_tolerance = float(ik_solver_properties.get('ik_position_tolerance', 0.01))
        use_orientation = bool(ik_solver_properties.get('use_orientation', False))

        _trace(f'[TRACE-Core] maxNumIterations = {max_num_iterations}')
        _trace(f'[TRACE-Core] residualThreshold = {residual_threshold}')
        _trace(f'[TRACE-Core] ik_position_tolerance = {ik_position_tolerance} m')
        _trace(f'[TRACE-Core] use_orientation = {use_orientation}')

        # --- DIAGNOSTIC: confirm link index used throughout ---
        _trace(f'[TRACE-Core-LINK] IK link index: {self.__tcp_link_index}  (name={self.__tcp_link_name})')
        try:
            link_name = pb.getJointInfo(self.__robot_id, self.__tcp_link_index)[12].decode('utf-8')
        except Exception:
            link_name = '<unknown>'
        _trace(f'[TRACE-Core-LINK] link index {self.__tcp_link_index} name: {link_name}')

        # --- Build null-space hints from robot parameters ---
        # These guide PyBullet toward valid solutions and damp oscillation.
        lower_limits = [float(lim[0]) for lim in self.__Robot_Parameters_Str.Theta.Limit]
        upper_limits = [float(lim[1]) for lim in self.__Robot_Parameters_Str.Theta.Limit]
        rest_poses   = [float(x)     for x  in self.__Robot_Parameters_Str.Theta.Home]
        joint_ranges = [upper - lower for lower, upper in zip(lower_limits, upper_limits)]
        joint_damping = [0.05] * len(self.__theta_index)

        if (len(lower_limits) == len(self.__theta_index) and
            len(upper_limits) == len(self.__theta_index) and
            len(rest_poses)   == len(self.__theta_index)):
            _trace(f'[TRACE-Core] Null-space hints available, passing to calculateInverseKinematics')
        else:
            _trace(f'[TRACE-Core] Length mismatch in null-space hints — skipping null-space params')

        # --- IK CALL ---
        _trace(f'[TRACE-Core] Calling pb.calculateInverseKinematics')
        _trace(f'[TRACE-Core]   link_index={self.__tcp_link_index} ({self.__tcp_link_name})')
        _trace(f'[TRACE-Core]   target_position = {target_position.tolist()}')
        _trace(f'[TRACE-Core]   maxNumIterations = {max_num_iterations}')
        _trace(f'[TRACE-Core]   residualThreshold = {residual_threshold}')
        _trace(f'[TRACE-Core]   use_orientation = {use_orientation}')

        if use_orientation:
            _trace(f'[TRACE-Core]   targetOrientation [x,y,z,w] = {target_orientation.tolist()}')
            theta_raw = pb.calculateInverseKinematics(
                self.__robot_id, self.__tcp_link_index,
                target_position.tolist(),
                targetOrientation=target_orientation.tolist(),
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                jointDamping=joint_damping,
                maxNumIterations=max_num_iterations,
                residualThreshold=residual_threshold,
            )
        else:
            _trace(f'[TRACE-Core]   targetOrientation = None (position-only IK)')
            theta_raw = pb.calculateInverseKinematics(
                self.__robot_id, self.__tcp_link_index,
                target_position.tolist(),
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                jointDamping=joint_damping,
                maxNumIterations=max_num_iterations,
                residualThreshold=residual_threshold,
            )
        _trace(f'[TRACE-Core] pb.calculateInverseKinematics RETURNED, type={type(theta_raw)}, '
               f'len={len(theta_raw) if hasattr(theta_raw, "__len__") else "N/A"}')
        _trace(f'[TRACE-Core] theta_raw (first 10) = {list(theta_raw[:10]) if hasattr(theta_raw, "__len__") else theta_raw}')

        # Truncate to the number of active joints.
        theta = np.array(theta_raw[:len(self.__theta_index)], dtype=np.float64)
        _trace(f'[TRACE-Core] theta (truncated to {len(theta)} values) = {theta}')

        # --- SUCCESS CHECKS (pre-ghost) ---
        _trace(f'[TRACE-Core] Running pre-ghost IK checks...')

        # Check 1: finite values
        if not np.all(np.isfinite(theta)):
            _trace(f'[TRACE-Core] FAIL: theta contains non-finite values: {theta}')
            successful = False
            out_of_limit = False
            ghost_tcp_pos = None
            position_error = None

        # Check 2: length matches
        elif len(theta) != len(self.__theta_index):
            _trace(f'[TRACE-Core] FAIL: theta length {len(theta)} != expected {len(self.__theta_index)}')
            successful = False
            out_of_limit = False
            ghost_tcp_pos = None
            position_error = None

        else:
            # Check 3: within joint limits — REJECT if any joint is out.
            _trace(f'[TRACE-Core] Checking joint limits...')
            out_of_limit = False
            for i, (th_i, th_limit) in enumerate(zip(theta, self.__Robot_Parameters_Str.Theta.Limit)):
                if not (th_limit[0] <= th_i <= th_limit[1]):
                    _trace(f'[TRACE-Core] Joint {i} out of limit: th_i={th_i:.6f}, limit={th_limit}')
                    out_of_limit = True

            if out_of_limit:
                _trace(f'[TRACE-Core] REJECT: at least one joint was out of limits — successful=False')
                successful = False
                ghost_tcp_pos = None
                position_error = None

            else:
                # All pre-ghost checks passed.  Validate on the ghost robot.
                _trace(f'[TRACE-Core] Pre-ghost checks passed. Validating on ghost robot...')

                for idx, th_index in enumerate(self.__theta_index):
                    pb.resetJointState(self.__robot_id_ghost, th_index, theta[idx])

                ghost_link_state = pb.getLinkState(
                    self.__robot_id_ghost, self.__ghost_tcp_link_index, computeForwardKinematics=True
                )
                ghost_tcp_pos = np.array(ghost_link_state[4])
                position_error = float(np.linalg.norm(target_position - ghost_tcp_pos))

                _trace(f'[TRACE-Core] Ghost TCP position = {ghost_tcp_pos}')
                _trace(f'[TRACE-Core] Target position   = {target_position}')
                _trace(f'[TRACE-Core] Position error     = {position_error:.6f} m  '
                       f'(tolerance = {ik_position_tolerance} m)')

                if position_error <= ik_position_tolerance:
                    successful = True
                    _trace(f'[TRACE-Core] PASS: position error {position_error:.6f} <= {ik_position_tolerance}')
                else:
                    successful = False
                    _trace(f'[TRACE-Core] FAIL: position error {position_error:.6f} > {ik_position_tolerance}')

        # --- GHOST VISUAL UPDATE ---
        _trace(f'[TRACE-Core] Updating ghost (successful={successful}, enable_ghost={enable_ghost})')
        if successful:
            self.__Reset_Ghost_Structure(theta, enable_ghost, [0.70, 0.85, 0.60])
        else:
            self.__Reset_Ghost_Structure(theta, enable_ghost, [0.85, 0.60, 0.60])

        # --- TRACE SUMMARY ---
        if ENABLE_TRACE:
            reason = []
            if successful:
                reason.append('PASS')
            else:
                if not np.all(np.isfinite(theta)):
                    reason.append('non-finite theta')
                elif len(theta) != len(self.__theta_index):
                    reason.append('theta length mismatch')
                elif out_of_limit:
                    reason.append('joint limit violation')
                else:
                    reason.append(f'position_error={position_error:.6f}m > {ik_position_tolerance}m')
            _trace(f'[TRACE-Core] FINAL: successful={successful}  reason={", ".join(reason)}  '
                   f'theta={theta.tolist()}')

        _trace(f'[TRACE-Core] Get_Inverse_Kinematics_Solution RETURNING '
               f'(successful={successful}, theta={theta.tolist()})')
        return (successful, theta)
