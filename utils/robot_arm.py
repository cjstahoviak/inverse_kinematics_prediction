import numpy as np

class RobotArm:
    def __init__(self, num_joints, joint_limits, link_lengths, rotation_axes, link_axes):
        """
        Initialize the robot arm.
        
        Parameters:
          num_joints (int): Number of revolute joints.
          joint_limits (list of tuples): Each tuple (min, max) defines the limits (radians) for a joint.
          link_lengths (list of floats): Lengths for each link; must have num_joints+1 elements.
          rotation_axes (list of str): Rotation axis for each joint. Each entry should be 'x', 'y', or 'z'.
                                       (Length must equal num_joints.)
          link_axes (list of str): Translation axis for each link. Each entry should be 'x', 'y', or 'z'.
                                   (Length must equal num_joints+1.)
        """
        self.num_joints = num_joints
        
        if len(joint_limits) != num_joints:
            raise ValueError("joint_limits must have length equal to num_joints.")
        self.joint_limits = joint_limits
        
        if len(link_lengths) != num_joints + 1:
            raise ValueError("link_lengths must have num_joints + 1 elements.")
        self.link_lengths = link_lengths
        
        if len(rotation_axes) != num_joints:
            raise ValueError("rotation_axes must have length equal to num_joints.")
        self.rotation_axes = rotation_axes
        
        if len(link_axes) != num_joints + 1:
            raise ValueError("link_axes must have length equal to num_joints + 1.")
        self.link_axes = link_axes

    @staticmethod
    def _rotx(theta):
        """Return a homogeneous rotation matrix about the x-axis by angle theta (radians)."""
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def _roty(theta):
        """Return a homogeneous rotation matrix about the y-axis by angle theta (radians)."""
        return np.array([
            [ np.cos(theta), 0, np.sin(theta), 0],
            [ 0,             1, 0,             0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [ 0,             0, 0,             1]
        ])

    @staticmethod
    def _rotz(theta):
        """Return a homogeneous rotation matrix about the z-axis by angle theta (radians)."""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0,              0,             1, 0],
            [0,              0,             0, 1]
        ])
    
    @staticmethod
    def _rot(axis, radians):
        """
        Return a homogeneous rotation matrix about the specified axis by angle in radians.
        """
        axis = axis.lower()
        if axis == 'x':
            return RobotArm._rotx(radians)
        elif axis == 'y':
            return RobotArm._roty(radians)
        elif axis == 'z':
            return RobotArm._rotz(radians)
        else:
            raise ValueError("Invalid rotation axis. Must be 'x', 'y', or 'z'.")

    @staticmethod
    def _transx(x):
        """Return a homogeneous translation matrix along the x-axis by distance x."""
        return np.array([
            [1, 0, 0, x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def _transy(y):
        """Return a homogeneous translation matrix along the y-axis by distance y."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    @staticmethod
    def _transz(z):
        """Return a homogeneous translation matrix along the z-axis by distance z."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def _trans(axis, distance):
        """
        Return a homogeneous translation matrix along the specified axis by distance.
        """
        axis = axis.lower()
        if axis == 'x':
            return RobotArm._transx(distance)
        elif axis == 'y':
            return RobotArm._transy(distance)
        elif axis == 'z':
            return RobotArm._transz(distance)
        else:
            raise ValueError("Invalid translation axis. Must be 'x', 'y', or 'z'.")

    @staticmethod
    def _rotation_matrix_to_quaternion(R):
        """
        Convert a 3x3 rotation matrix R to a quaternion.
        Returns a numpy array in the order [w, x, y, z].
        """
        q = np.empty(4)
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2  # S = 4*w
            q[0] = 0.25 * S
            q[1] = (R[2, 1] - R[1, 2]) / S
            q[2] = (R[0, 2] - R[2, 0]) / S
            q[3] = (R[1, 0] - R[0, 1]) / S
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4*qx
                q[0] = (R[2, 1] - R[1, 2]) / S
                q[1] = 0.25 * S
                q[2] = (R[0, 1] + R[1, 0]) / S
                q[3] = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4*qy
                q[0] = (R[0, 2] - R[2, 0]) / S
                q[1] = (R[0, 1] + R[1, 0]) / S
                q[2] = 0.25 * S
                q[3] = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4*qz
                q[0] = (R[1, 0] - R[0, 1]) / S
                q[1] = (R[0, 2] + R[2, 0]) / S
                q[2] = (R[1, 2] + R[2, 1]) / S
                q[3] = 0.25 * S
        return q
    
    @staticmethod
    def _quaternion_to_rotation_matrix(quat):
        """
        Convert a quaternion (in ROS format: (x, y, z, w)) to a 3x3 rotation matrix.
        """
        x, y, z, w = quat
        R = np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])
        return R

    def forward_kinematics(self, joint_angles):
        """
        Compute the forward kinematics for the robot arm.
        
        For each joint i, apply a translation along the specified link axis (from self.link_axes)
        using the corresponding link length, then apply a rotation about the specified axis
        (from self.rotation_axes) using the corresponding joint angle. After processing all joints,
        apply a final translation for the end-effector link.
        
        Parameters:
        joint_angles (list or array): Joint angles (in radians) of length equal to num_joints.
        
        Returns:
        position (numpy array): [x, y, z] position of the end effector.
        quaternion (tuple): Orientation of the end effector as (x, y, z, w) in ROS format.
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError("Length of joint_angles must equal the number of joints.")
        
        T = np.eye(4)
        
        # Translate along the first link (base to first joint)
        axis_link = self.link_axes[0].lower()
        length = self.link_lengths[0]
        T = T.dot(self._trans(axis_link, length))
        
        # Process each joint
        for i in range(self.num_joints):
            # Apply joint rotation
            axis_rot = self.rotation_axes[i].lower()
            theta = joint_angles[i]
            T = T.dot(self._rot(axis_rot, theta))
            
            # If not the last joint, translate to the next joint
            if i < self.num_joints - 1:
                axis_link = self.link_axes[i+1].lower()
                length = self.link_lengths[i+1]
                T = T.dot(self._trans(axis_link, length))
        
        # Apply final translation for the end-effector link
        final_axis = self.link_axes[-1].lower()
        T = T.dot(self._trans(final_axis, self.link_lengths[-1]))
        
        position = T[0:3, 3]
        R = T[0:3, 0:3]
        quat = self._rotation_matrix_to_quaternion(R)
        # Convert from [w, x, y, z] to ROS standard (x, y, z, w)
        quaternion_ros = (quat[1], quat[2], quat[3], quat[0])
        return position, quaternion_ros
    
    def get_joint_poses(self, joint_angles):
        """
        Compute and return the 3D poses (position and orientation) of the base, each intermediate joint,
        and the end-effector for the given joint angles.
        
        Parameters:
        joint_angles (list or np.ndarray): Joint angles (in radians) with length equal to num_joints.
        
        Returns:
        poses (list of tuples): Each element is a tuple (position, quaternion), where:
            - position is an np.ndarray of shape (3,) representing the 3D coordinates.
            - quaternion is a tuple (x, y, z, w) representing the orientation in ROS format.
            The list includes the base, each joint, and the end-effector.
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError("Length of joint_angles must equal the number of joints.")
        
        poses = []
        # Start with the base pose (T is identity).
        T = np.eye(4)
        base_pos = T[:3, 3].copy()
        base_R = T[:3, :3]
        base_quat = self._rotation_matrix_to_quaternion(base_R)
        base_quat_ros = (base_quat[1], base_quat[2], base_quat[3], base_quat[0])
        poses.append((base_pos, base_quat_ros))
        
        # Translate to the first joint position
        axis_link = self.link_axes[0].lower()
        T = T.dot(self._trans(axis_link, self.link_lengths[0]))
        
        # Record the position of the first joint (before any rotation)
        first_joint_pos = T[:3, 3].copy()
        first_joint_R = T[:3, :3]
        first_joint_quat = self._rotation_matrix_to_quaternion(first_joint_R)
        first_joint_quat_ros = (first_joint_quat[1], first_joint_quat[2], first_joint_quat[3], first_joint_quat[0])
        poses.append((first_joint_pos, first_joint_quat_ros))
        
        # Process each joint and subsequent link.
        for i in range(self.num_joints):
            # Apply joint rotation.
            axis_rot = self.rotation_axes[i].lower()
            theta = joint_angles[i]
            T = T.dot(self._rot(axis_rot, theta))
            
            # If not the last joint, translate to the next joint
            if i < self.num_joints - 1:
                axis_link = self.link_axes[i+1].lower()
                T = T.dot(self._trans(axis_link, self.link_lengths[i+1]))
                
                # Record the pose at this joint.
                pos = T[:3, 3].copy()
                R = T[:3, :3]
                quat = self._rotation_matrix_to_quaternion(R)
                quat_ros = (quat[1], quat[2], quat[3], quat[0])
                poses.append((pos, quat_ros))
        
        # Apply the final translation for the end-effector.
        final_axis = self.link_axes[-1].lower()
        T = T.dot(self._trans(final_axis, self.link_lengths[-1]))
        pos = T[:3, 3].copy()
        R = T[:3, :3]
        quat = self._rotation_matrix_to_quaternion(R)
        quat_ros = (quat[1], quat[2], quat[3], quat[0])
        poses.append((pos, quat_ros))
        
        return poses
    
    def sample_random_joint_angles(self):
        """
        Return a random set of joint angles within their respective limits.
        
        Returns:
          np.ndarray: An array of random joint angles with shape (num_joints,).
        """
        return np.array([np.random.uniform(low, high) for (low, high) in self.joint_limits])
