# reference.py
import numpy as np
import config

class ReferenceTrajectory:
    def __init__(self, N, dt=config.DT):
        self.N = N
        self.dt = dt

    def get_reference(self, current_s, target_lane, target_vel):
        """
        Generates a kinematically feasible reference trajectory (Eq. 4).
        Returns arrays for x_ref and u_ref over the horizon N.
        """
        x_ref = np.zeros((self.N + 1, 4)) # [s, e_y, e_psi, v]
        u_ref = np.zeros((self.N, 2))     # [a, psi_dot]
        
        # Start the reference at the EV's current longitudinal progress
        x_ref[0] = [current_s, target_lane, 0.0, target_vel]
        
        for k in range(self.N):
            # Constant velocity, straight line forward
            s_next = x_ref[k, 0] + target_vel * self.dt
            x_ref[k+1] = [s_next, target_lane, 0.0, target_vel]
            
            # Inputs required to maintain this reference (zero accel, zero steering)
            u_ref[k] = [0.0, 0.0] 
            
        return x_ref, u_ref