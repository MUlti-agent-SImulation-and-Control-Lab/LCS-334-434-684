import numpy as np
import math
from ugv_uav_control.view_quality import compute_view_quality

def wrap_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def process_model(state, u, dt, alpha_v, alpha_w):
    """
    Unicycle process model with first-order actuator dynamics.
    state: [x, y, theta, v, omega]
    u: [v_cmd, omega_cmd]
    """
    x, y, theta, v, omega = state
    v_cmd, omega_cmd = u
    
    # Actuator dynamics
    v_next = v + alpha_v * (v_cmd - v) * dt
    omega_next = omega + alpha_w * (omega_cmd - omega) * dt
    
    # Average velocities for trapezoidal integration
    v_avg = (v + v_next) / 2.0
    omega_avg = (omega + omega_next) / 2.0
    
    # Kinematics
    if abs(omega_avg) < 1e-4:
        x_next = x + v_avg * np.cos(theta) * dt
        y_next = y + v_avg * np.sin(theta) * dt
    else:
        x_next = x + (v_avg / omega_avg) * (np.sin(theta + omega_avg * dt) - np.sin(theta))
        y_next = y - (v_avg / omega_avg) * (np.cos(theta + omega_avg * dt) - np.cos(theta))
    
    theta_next = wrap_angle(theta + omega_avg * dt)
    
    return np.array([x_next, y_next, theta_next, v_next, omega_next])

def compute_jacobian_F(state, dt, alpha_v, alpha_w):
    """Compute state transition Jacobian F = d(f)/dx."""
    x, y, theta, v, omega = state
    
    F = np.eye(5)
    F[0, 2] = -v * np.sin(theta) * dt
    F[0, 3] = np.cos(theta) * dt
    F[1, 2] = v * np.cos(theta) * dt
    F[1, 3] = np.sin(theta) * dt
    F[2, 4] = dt
    F[3, 3] = 1.0 - alpha_v * dt
    F[4, 4] = 1.0 - alpha_w * dt
    
    return F

def kalman_update(Sigma, H, R):
    """
    Joseph-form EKF covariance update for numerical stability.
    Sigma: predicted covariance (5x5)
    H: measurement Jacobian (mx5)
    R: measurement noise covariance (mxm)
    """
    S = H @ Sigma @ H.T + R
    # Inversion safety Check
    K = Sigma @ H.T @ np.linalg.inv(S)
    
    I_KH = np.eye(Sigma.shape[0]) - K @ H
    Sigma_updated = I_KH @ Sigma @ I_KH.T + K @ R @ K.T
    
    # Final symmetry enforcement
    Sigma_updated = (Sigma_updated + Sigma_updated.T) / 2.0
    return Sigma_updated

def get_dynamic_R(uav_pos, ugv_pos_world, R_base, sharpness=5.0):
    """
    Compute dynamic measurement noise based on UAV-UGV view quality.
    uav_pos: [x, y, z]
    ugv_pos_world: [x, y]
    """
    q_k = compute_view_quality(
        uav_pos[0], uav_pos[1], uav_pos[2],
        ugv_pos_world[0], ugv_pos_world[1],
        sharpness=sharpness
    )
    # Scale R inversely with quality (add epsilon to avoid division by zero)
    r_scale = 1.0 / (q_k + 1e-3)
    
    if np.isscalar(R_base):
        # Default to 2x2 for [x, y] observation if scalar is provided
        return np.eye(2) * (R_base * r_scale), q_k
    else:
        # Scale the provided matrix (robust to any shape)
        return R_base * r_scale, q_k
