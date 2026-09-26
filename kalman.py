import numpy as np
import math

class CircularKalmanFilter:
    def __init__(self, initial_theta, t_scan, sigma_a=0.1, sigma_z=0.05):
        # State vector [theta (rad), omega (rad/s)]
        initial_omega = 2.0 * np.pi / t_scan
        self.x = np.array([[initial_theta],
                           [initial_omega]], dtype=np.float64)
       
        # P: Covariance Matrix (start with high uncertainty)
        self.P = np.array([[np.pi, 0.0],
                           [0.0, 1.0]], dtype=np.float64)
       
        # H: Observation Matrix (we only measure angle, not speed)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
       
        # R: Measurement Noise Matrix
        self.R = np.array([[sigma_z**2]], dtype=np.float64)
       
        # Process noise variance (acceleration jitter)
        self.sigma_a2 = sigma_a**2

    def predict(self, dt):
        # F: State Transition Matrix
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float64)
       
        # Q: Process Noise Matrix (Discrete White Noise Kinematic Model)
        Q = np.array([[(dt**3) / 3.0, (dt**2) / 2.0],
                      [(dt**2) / 2.0, dt]], dtype=np.float64) * self.sigma_a2
                     
        # Predict forward
        self.x = F @ self.x
       
        # Keep internal state phase wrapped [0, 2pi]
        self.x[0, 0] = self.x[0, 0] % (2.0 * np.pi)
       
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_theta):
        # Calculate residual using trigonometric phase wrapping (The 360-deg fix)
        predicted_theta = (self.H @ self.x)[0, 0]
        y_raw = measured_theta - predicted_theta
        y_wrapped = math.atan2(math.sin(y_raw), math.cos(y_raw))
       
        # Calculate Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
       
        # Update state and shrink covariance
        self.x = self.x + K * y_wrapped
        self.x[0, 0] = self.x[0, 0] % (2.0 * np.pi)
       
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P