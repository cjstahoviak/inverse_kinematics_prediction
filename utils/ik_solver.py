# ik_solver.py
import numpy as np
import abc
import time

class IKSolver(abc.ABC):
    """Abstract base class for inverse kinematics solvers."""
    
    def __init__(self, robot_arm):
        """
        Initialize the IK solver with a reference to the robot arm.
        
        Parameters:
            robot_arm: A RobotArm instance.
        """
        self.robot_arm = robot_arm
    
    @abc.abstractmethod
    def solve(self, target_position, target_orientation=None, initial_joint_angles=None):
        """
        Solve the inverse kinematics problem.
        
        Parameters:
            target_position: Target position for the end-effector.
            target_orientation: Optional target orientation.
            initial_joint_angles: Optional initial guess for joint angles.
            
        Returns:
            (joint_angles, computation_time, success_flag)
        """
        pass
    
    def _quaternion_difference(self, q1, q2):
        """
        Compute the difference between two quaternions in terms of a 3D rotation vector.
        
        Parameters:
            q1 (tuple): First quaternion in (x, y, z, w) format.
            q2 (tuple): Second quaternion in (x, y, z, w) format.
        
        Returns:
            np.ndarray: 3D rotation vector representing the rotation from q1 to q2.
        """
        # Convert quaternions to rotation matrices
        R1 = self.robot_arm._quaternion_to_rotation_matrix(q1)
        R2 = self.robot_arm._quaternion_to_rotation_matrix(q2)
        
        # Compute the rotation matrix that transforms from R1 to R2
        R_diff = R2.dot(R1.T)
        
        # Convert the rotation matrix to axis-angle representation
        angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_diff) - 1) / 2)))  # Clamp to avoid numerical issues
        
        if np.abs(angle) < 1e-10:
            # If angle is very small, return zero vector
            return np.zeros(3)
        else:
            # Calculate rotation axis
            axis = np.array([
                R_diff[2, 1] - R_diff[1, 2],
                R_diff[0, 2] - R_diff[2, 0],
                R_diff[1, 0] - R_diff[0, 1]
            ])
            axis = axis / (2 * np.sin(angle))
            
            # Return the rotation vector
            return axis * angle

    def _quaternion_differential(self, q1, q2, delta):
        """
        Compute the differential rotation between two quaternions.
        
        Parameters:
            q1 (tuple): First quaternion in (x, y, z, w) format.
            q2 (tuple): Second quaternion in (x, y, z, w) format.
            delta (float): The time or parameter difference used for differentiation.
        
        Returns:
            np.ndarray: 3D angular velocity vector.
        """
        # Compute the quaternion difference
        diff = self._quaternion_difference(q1, q2)
        
        # Divide by delta to get the differential
        return diff / delta
    
    def _compute_jacobian(self, joint_angles, include_orientation=False):
        """
        Compute the Jacobian matrix for the current joint configuration.
        
        Parameters:
            joint_angles: Current joint angles.
            include_orientation: Whether to include orientation rows in the Jacobian.
            
        Returns:
            The Jacobian matrix.
        """
        delta = 1e-6
        output_dims = 6 if include_orientation else 3
        jacobian = np.zeros((output_dims, self.robot_arm.num_joints))
        
        current_position, current_orientation = self.robot_arm.forward_kinematics(joint_angles)
        
        for i in range(self.robot_arm.num_joints):
            joint_angles_plus = joint_angles.copy()
            joint_angles_plus[i] += delta
            
            position_plus, orientation_plus = self.robot_arm.forward_kinematics(joint_angles_plus)
            
            position_diff = (position_plus - current_position) / delta
            jacobian[0:3, i] = position_diff
            
            if include_orientation:
                orientation_diff = self._quaternion_differential(current_orientation, orientation_plus, delta)
                jacobian[3:6, i] = orientation_diff
        
        return jacobian


class DampedLeastSquaresSolver(IKSolver):
    """Inverse kinematics solver using the Damped Least Squares method."""
    
    def __init__(self, robot_arm, damping_factor=0.1, max_iterations=100, 
                 position_tolerance=0.001, orientation_tolerance=0.01):
        """
        Initialize the DLS solver.
        
        Parameters:
            robot_arm: A RobotArm instance.
            damping_factor: Damping factor (lambda) for the DLS method.
            max_iterations: Maximum number of iterations.
            position_tolerance: Convergence threshold for position.
            orientation_tolerance: Convergence threshold for orientation.
        """
        super().__init__(robot_arm)
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
    
    def solve(self, target_position, target_orientation=None, initial_joint_angles=None):
        """
        Solve IK using the Damped Least Squares method.
        
        Parameters and return values as defined in the base class.
        """
        start_time = time.time()
        
        # Initialize joint angles
        if initial_joint_angles is None:
            current_joint_angles = self.robot_arm.sample_random_joint_angles()
        else:
            current_joint_angles = np.array(initial_joint_angles)
        
        # Convert target position to numpy array
        target_position = np.array(target_position)
        
        # Main DLS iteration loop
        for iteration in range(self.max_iterations):
            # Forward kinematics to get current position and orientation
            current_position, current_orientation = self.robot_arm.forward_kinematics(current_joint_angles)
            
            # Calculate position error
            position_error = target_position - current_position
            position_error_norm = np.linalg.norm(position_error)
            
            # Calculate orientation error if target orientation is provided
            if target_orientation is not None:
                orientation_error = self._quaternion_difference(current_orientation, target_orientation)
                orientation_error_norm = np.linalg.norm(orientation_error)
                # Combine errors
                error = np.concatenate([position_error, orientation_error])
            else:
                error = position_error
                orientation_error_norm = 0
            
            # Check convergence
            if position_error_norm < self.position_tolerance and (target_orientation is None or 
                                                                 orientation_error_norm < self.orientation_tolerance):
                computation_time = time.time() - start_time
                return current_joint_angles, computation_time, True
            
            # Compute the Jacobian matrix
            jacobian = self._compute_jacobian(current_joint_angles, target_orientation is not None)
            
            # Apply DLS formula: Δθ = J^T (J J^T + λ^2 I)^(-1) e
            identity = np.eye(jacobian.shape[0])
            lambda_squared = self.damping_factor ** 2
            jacobian_transpose = jacobian.T
            
            temp = jacobian.dot(jacobian_transpose) + lambda_squared * identity
            temp = np.linalg.solve(temp, error)
            delta_theta = jacobian_transpose.dot(temp)
            
            # Update joint angles
            current_joint_angles = current_joint_angles + delta_theta
            
            # Enforce joint limits
            for i in range(self.robot_arm.num_joints):
                low, high = self.robot_arm.joint_limits[i]
                current_joint_angles[i] = max(low, min(high, current_joint_angles[i]))
        
        # If we reached max iterations without converging
        computation_time = time.time() - start_time
        return current_joint_angles, computation_time, False


class JacobianTransposeSolver(IKSolver):
    """Inverse kinematics solver using the Jacobian Transpose method."""
    
    def __init__(self, robot_arm, step_size=0.1, max_iterations=100, 
                 position_tolerance=0.001, orientation_tolerance=0.01):
        """
        Initialize the Jacobian Transpose solver.
        
        Parameters:
            robot_arm: A RobotArm instance.
            step_size: Step size for the gradient descent.
            max_iterations: Maximum number of iterations.
            position_tolerance: Convergence threshold for position.
            orientation_tolerance: Convergence threshold for orientation.
        """
        super().__init__(robot_arm)
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
    
    def solve(self, target_position, target_orientation=None, initial_joint_angles=None):
        """
        Solve IK using the Jacobian Transpose method.
        
        Parameters and return values as defined in the base class.
        """
        start_time = time.time()
        
        # Initialize joint angles
        if initial_joint_angles is None:
            current_joint_angles = self.robot_arm.sample_random_joint_angles()
        else:
            current_joint_angles = np.array(initial_joint_angles)
        
        # Convert target position to numpy array
        target_position = np.array(target_position)
        
        # Main iteration loop
        for iteration in range(self.max_iterations):
            # Forward kinematics to get current position and orientation
            current_position, current_orientation = self.robot_arm.forward_kinematics(current_joint_angles)
            
            # Calculate position error
            position_error = target_position - current_position
            position_error_norm = np.linalg.norm(position_error)
            
            # Calculate orientation error if target orientation is provided
            if target_orientation is not None:
                orientation_error = self._quaternion_difference(current_orientation, target_orientation)
                orientation_error_norm = np.linalg.norm(orientation_error)
                # Combine errors
                error = np.concatenate([position_error, orientation_error])
            else:
                error = position_error
                orientation_error_norm = 0
            
            # Check convergence
            if position_error_norm < self.position_tolerance and (target_orientation is None or 
                                                                 orientation_error_norm < self.orientation_tolerance):
                computation_time = time.time() - start_time
                return current_joint_angles, computation_time, True
            
            # Compute the Jacobian matrix
            jacobian = self._compute_jacobian(current_joint_angles, target_orientation is not None)
            
            # Apply Jacobian Transpose formula: Δθ = α * J^T * e
            # Where α is the step size
            delta_theta = self.step_size * jacobian.T.dot(error)
            
            # Update joint angles
            current_joint_angles = current_joint_angles + delta_theta
            
            # Enforce joint limits
            for i in range(self.robot_arm.num_joints):
                low, high = self.robot_arm.joint_limits[i]
                current_joint_angles[i] = max(low, min(high, current_joint_angles[i]))
        
        # If we reached max iterations without converging
        computation_time = time.time() - start_time
        return current_joint_angles, computation_time, False


class PseudoInverseSolver(IKSolver):
    """Inverse kinematics solver using the Pseudoinverse method."""
    
    def __init__(self, robot_arm, max_iterations=100, 
                 position_tolerance=0.001, orientation_tolerance=0.01):
        super().__init__(robot_arm)
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
    
    def solve(self, target_position, target_orientation=None, initial_joint_angles=None):
        """Implement the Pseudoinverse method here"""
        start_time = time.time()
        
        # Initialize joint angles
        if initial_joint_angles is None:
            current_joint_angles = self.robot_arm.sample_random_joint_angles()
        else:
            current_joint_angles = np.array(initial_joint_angles)
        
        # Convert target position to numpy array
        target_position = np.array(target_position)
        
        # Main iteration loop
        for iteration in range(self.max_iterations):
            # Forward kinematics
            current_position, current_orientation = self.robot_arm.forward_kinematics(current_joint_angles)
            
            # Calculate errors
            position_error = target_position - current_position
            position_error_norm = np.linalg.norm(position_error)
            
            if target_orientation is not None:
                orientation_error = self._quaternion_difference(current_orientation, target_orientation)
                orientation_error_norm = np.linalg.norm(orientation_error)
                error = np.concatenate([position_error, orientation_error])
            else:
                error = position_error
                orientation_error_norm = 0
            
            # Check convergence
            if position_error_norm < self.position_tolerance and (target_orientation is None or 
                                                                orientation_error_norm < self.orientation_tolerance):
                computation_time = time.time() - start_time
                return current_joint_angles, computation_time, True
            
            # Compute Jacobian
            jacobian = self._compute_jacobian(current_joint_angles, target_orientation is not None)
            
            # Compute pseudoinverse: J+ = (J^T J)^(-1) J^T
            # Be careful with singular matrices
            try:
                # Compute J^T J
                j_transpose_j = jacobian.T.dot(jacobian)
                # Compute (J^T J)^(-1)
                j_transpose_j_inv = np.linalg.inv(j_transpose_j)
                # Compute (J^T J)^(-1) J^T
                pseudoinverse = j_transpose_j_inv.dot(jacobian.T)
                # Compute delta_theta = J+ * error
                delta_theta = pseudoinverse.dot(error)
            except np.linalg.LinAlgError:
                # If matrix is singular, use a small random step
                delta_theta = np.random.normal(0, 0.01, self.robot_arm.num_joints)
            
            # Update joint angles
            current_joint_angles = current_joint_angles + delta_theta
            
            # Enforce joint limits
            for i in range(self.robot_arm.num_joints):
                low, high = self.robot_arm.joint_limits[i]
                current_joint_angles[i] = max(low, min(high, current_joint_angles[i]))
        
        computation_time = time.time() - start_time
        return current_joint_angles, computation_time, False