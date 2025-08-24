"""
Kalman Filter for object tracking in PixelFlow.

This module implements a Kalman filter for tracking bounding boxes in image space.
The filter uses an 8-dimensional state space (x, y, aspect_ratio, height, vx, vy, va, vh)
to track object position, size, and velocity.
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    A Kalman filter for tracking bounding boxes in image space.
    
    The 8-dimensional state space:
        x, y, a, h, vx, vy, va, vh
    
    Contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space.
    """
    
    def __init__(self):
        ndim, dt = 4, 1.0
        
        # Create motion and update matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Standard deviation weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
    
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create track from an unassociated measurement.
        
        Args:
            measurement: Bounding box coordinates (x, y, a, h) with center position (x, y),
                        aspect ratio a, and height h.
        
        Returns:
            tuple: (mean, covariance) - 8-dimensional mean vector and 8x8 covariance matrix
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        
        Args:
            mean: 8-dimensional mean vector of the object state at previous time step
            covariance: 8x8 covariance matrix of the object state at previous time step
        
        Returns:
            tuple: (mean, covariance) - Predicted mean vector and covariance matrix
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        
        return mean, covariance
    
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        
        Args:
            mean: 8-dimensional state mean vector
            covariance: 8x8 state covariance matrix
        
        Returns:
            tuple: (mean, covariance) - Projected mean and covariance in measurement space
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, 
               measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        
        Args:
            mean: 8-dimensional predicted state mean vector
            covariance: 8x8 predicted state covariance matrix
            measurement: 4-dimensional measurement vector (x, y, a, h)
        
        Returns:
            tuple: (mean, covariance) - Updated state distribution
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance
    
    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                       measurements: np.ndarray, only_position: bool = False) -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        
        Args:
            mean: 8-dimensional state mean vector
            covariance: 8x8 state covariance matrix
            measurements: Nx4 measurement matrix
            only_position: If True, only use position for distance calculation
        
        Returns:
            np.ndarray: Array of gating distances
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha