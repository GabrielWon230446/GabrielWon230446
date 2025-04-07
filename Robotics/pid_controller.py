import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, accuracy_requirement=0.001):
        """
        Initialize the PID controller with given parameters.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            accuracy_requirement (float): The accuracy threshold for distance (meters).
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Initialize error terms
        self.prev_error = None
        self.integral = 0

    def compute_action(self, current_position, goal_position, dt):
        """
        Compute the action based on the PID formula.

        Args:
            current_position (np.ndarray): Current pipette position (x, y, z).
            goal_position (np.ndarray): Target goal position (x, y, z).
            dt (float): Time step between updates.

        Returns:
            np.ndarray: PID computed action (x, y, z).
        """
        error = goal_position - current_position
        distance = np.linalg.norm(error)

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = np.zeros_like(error)
        if self.prev_error is not None:
            derivative = self.kd * (error - self.prev_error) / dt

        # Update previous error
        self.prev_error = error

        # Compute PID action
        action = proportional + integral + derivative
        return np.clip(action, -1, 1)  # Clip the action to [-1, 1]

