# dynamics.py
import numpy as np
import cartersian_coordinates.config as config

class Vehicle:
    def __init__(self, x, y, v, vehicle_id="target"):
        """
        State: [x, y, v]
        """
        self.x = x
        self.y = y
        self.v = v
        self.id = vehicle_id

    def step(self, a, lane_change_rate, dt=config.DT):
        """
        Updates the vehicle state using explicit Euler integration.
        a: acceleration (longitudinal)
        lane_change_rate: lateral velocity (dy/dt)
        """
        # Update velocity and clip to realistic bounds
        self.v = self.v + a * dt
        self.v = np.clip(self.v, config.MIN_VELOCITY, config.MAX_VELOCITY)
        
        # Update position
        self.x = self.x + self.v * dt
        self.y = self.y + lane_change_rate * dt

    def get_state(self):
        return np.array([self.x, self.y, self.v])