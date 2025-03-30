import numpy as np

class RobotArm:
    def __init__(self, num_joints, joint_limits, link_lengths):
        """
        Initialize the robot arm.
        
        Parameters:
          num_joints (int): Number of revolute joints.
          joint_limits (list of tuples): Each tuple (min, max) defines the limits (radians) for a joint.
          link_lengths (list of floats): Lengths for each link; must have num_joints+1 elements.
        """
        self.num_joints = num_joints
        if len(joint_limits) != num_joints:
            raise ValueError("joint_limits must have length equal to num_joints.")
        self.joint_limits = joint_limits
        
        if len(link_lengths) != num_joints + 1:
            raise ValueError("link_lengths must have num_joints + 1 elements.")
        self.link_lengths = link_lengths

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
    def _roty(theta):
        """Return a homogeneous rotation matrix about the y-axis by angle theta (radians)."""
        return np.array([
            [ np.cos(theta), 0, np.sin(theta), 0],
            [ 0,             1, 0,             0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [ 0,             0, 0,             1]
        ])
    
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

    def forward_kinematics(self, joint_angles):
        """
        Compute the forward kinematics for the robot arm.
        
        The computation follows a chain of transformations:
          - For the first joint, a rotation about the z-axis is applied.
          - For joints 2 through n, rotations about the y-axis are applied.
          - After each joint rotation, a translation along the x-axis is applied using the corresponding link length.
          - A final translation using the last link length is applied for the end-effector.
        
        Parameters:
          joint_angles (list or array): Joint angles (in radians) of length equal to num_joints.
        
        Returns:
          position (numpy array): [x, y, z] position of the end effector.
          quaternion (tuple): Orientation of the end effector as (x, y, z, w) in ROS format.
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError("Length of joint_angles must equal the number of joints.")
        
        # Initialize the transformation as the identity matrix.
        T = np.eye(4)
        
        # Apply transformation for the first joint (rotate about z then translate)
        T = T.dot(self._rotz(joint_angles[0])).dot(self._transx(self.link_lengths[0]))
        
        # Apply transformation for remaining joints (rotate about y then translate)
        for i in range(1, self.num_joints):
            T = T.dot(self._roty(joint_angles[i])).dot(self._transx(self.link_lengths[i]))
        
        # Apply the final translation for the end-effector link.
        T = T.dot(self._transx(self.link_lengths[-1]))
        
        # Extract the position (translation part) from the transformation matrix.
        position = T[0:3, 3]
        
        # Extract the rotation matrix and convert it to a quaternion.
        R = T[0:3, 0:3]
        quat = self._rotation_matrix_to_quaternion(R)
        # Reorder to ROS standard (x, y, z, w)
        quaternion_ros = (quat[1], quat[2], quat[3], quat[0])
        
        return position, quaternion_ros
