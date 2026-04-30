# Functionality: This file implements the Non-Linear Kinematic Bicycle Model in the Frenet frame,
#  which is exactly what the authors use in Equation (1) of the paper. 
# By tracking state variables like arc length ($s$), lateral offset ($e_y$), 
# relative heading ($e_\psi$), and speed ($v$), the controller doesn't need to worry 
# about the complex X/Y curves of the track


import numpy as np
import config

class FrenetVehicle:
    def __init__(self, id_name, s_init, ey_init, v_init, color):
        self.id = id_name
        self.color = color
        
        # State Vector: [s, ey, epsi, v]
        self.s = s_init
        self.ey = ey_init
        self.epsi = 0.0  # Relative heading to the lane (starts perfectly aligned)
        self.v = v_init
        
    def get_curvature(self, s):
        """Returns track curvature kappa(s) at current arc length."""
        s = s % config.TRACK_LENGTH
        if s < config.S1:
            return 0.0  # Straight
        elif s < config.S2:
            return 1.0 / config.CURVE_RADIUS  # Curve
        elif s < config.S3:
            return 0.0  # Straight
        else:
            return 1.0 / config.CURVE_RADIUS  # Curve
            
    def update(self, a, yaw_rate, dt):
        """
        Updates the state using the non-linear Frenet kinematic equations 
        from the SMPC paper.
        Inputs: a (acceleration), yaw_rate (global yaw rate)
        """
        kappa = self.get_curvature(self.s)
        
        # Prevent division by zero if vehicle drifts too far inside the curve
        denominator = 1.0 - self.ey * kappa
        if denominator < 0.1: denominator = 0.1 
        
        # Frenet Kinematic Equations
        s_dot = (self.v * np.cos(self.epsi)) / denominator
        ey_dot = self.v * np.sin(self.epsi)
        epsi_dot = yaw_rate - (s_dot * kappa)
        v_dot = a
        
        # Forward Euler Integration
        self.s = (self.s + s_dot * dt) % config.TRACK_LENGTH
        self.ey += ey_dot * dt
        self.epsi += epsi_dot * dt
        self.v += v_dot * dt
        
        # Keep relative heading bounded between -pi and pi
        self.epsi = (self.epsi + np.pi) % (2 * np.pi) - np.pi


    # We need to give our dummy vehicles the ability to "predict" their future so the Ego vehicle can read it.
    def get_prediction(self, N, dt):
        """
        Generates a deterministic constant-velocity prediction 
        for the Target Vehicle over the horizon N.
        Returns: numpy array of shape (N, 4) containing [s, ey, epsi, v]
        """
        prediction = np.zeros((N, 4))
        current_s = self.s
        
        for k in range(N):
            # Predict future s position based on current velocity
            next_s = (current_s + self.v * dt) % config.TRACK_LENGTH
            prediction[k, :] = [next_s, self.ey, self.epsi, self.v]
            current_s = next_s
            
        return prediction