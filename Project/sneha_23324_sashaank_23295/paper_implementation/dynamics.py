<<<<<<< HEAD
# dynamics.py
import numpy as np
import config

class EgoVehicle:
    def __init__(self, s, e_y, e_psi, v):
        """
        State x_t: [s, e_y, e_psi, v]
        s: arc length (longitudinal progress)
        e_y: lateral error from centerline (y=0)
        e_psi: heading error relative to centerline
        v: velocity
        """
        self.state = np.array([s, e_y, e_psi, v], dtype=float)

    def step(self, action, dt=config.DT):
        """
        action u_t: [a, psi_dot] (acceleration, yaw rate)
        Euler discretization of the Frenet kinematic  model (Eq. 1 in paper).
        Assuming straight road curvature (kappa = 0).
        """
        a, psi_dot = action
        s, e_y, e_psi, v = self.state

        # Derivatives
        s_dot = v * np.cos(e_psi)
        e_y_dot = v * np.sin(e_psi)
        e_psi_dot = psi_dot
        v_dot = a

        # Update state
        self.state[0] += s_dot * dt
        self.state[1] += e_y_dot * dt
        self.state[2] += e_psi_dot * dt
        self.state[3] += v_dot * dt

        # Clip limits
        self.state[3] = np.clip(self.state[3], config.MIN_VELOCITY, config.MAX_VELOCITY)
        
    def get_state(self):
        return self.state.copy()

class TargetVehicle:
    def __init__(self, x, y, vx, tv_id):
        """
        Target Vehicle modeled in Cartesian coordinates.
        State: [X, Y, v_x, v_y]
        """
        self.state = np.array([x, y, vx, 0.0], dtype=float)
        self.id = tv_id

    def step(self, mode, dt=config.DT):
        """
        LTV update based on the designated maneuver mode.
        """
        x, y, vx, vy = self.state

        if mode == config.MODE_KEEP_LANE:
            # Maintain current lane and speed
            vy_new = 0.0
            vx_new = vx 
        elif mode == config.MODE_CHANGE_LEFT:
            # Move laterally to the left
            vy_new = 1.0  # m/s lateral speed
            vx_new = vx
        elif mode == config.MODE_CHANGE_RIGHT:
            vy_new = -1.0
            vx_new = vx
        else:
            vy_new, vx_new = 0.0, vx

        # Simple kinematic update
        self.state[0] += vx_new * dt
        self.state[1] += vy_new * dt
        self.state[2] = vx_new
        self.state[3] = vy_new

    def get_state(self):
        return self.state.copy()
    
    def get_multimodal_predictions(self, N, dt=config.DT):
        """
        Generates open-loop trajectory predictions for the MPC to evaluate.
        Returns a dictionary of modes containing:
        - 'prob': Probability of this mode occurring
        - 'traj': Array of predicted [X, Y] states over horizon N
        - 'cov': Array of spatial uncertainties (variances) over horizon N
        """
        predictions = {}
        x, y, vx, vy = self.state

        # Mode 0: Keep Lane (High Probability)
        prob_keep = 0.8
        traj_keep = np.zeros((N, 2))
        cov_keep = np.zeros(N) # Uncertainty grows over time
        
        for k in range(N):
            traj_keep[k] = [x + vx * (k * dt), y]
            cov_keep[k] = 0.1 * k # Variance grows linearly with time
            
        predictions[config.MODE_KEEP_LANE] = {
            'prob': prob_keep, 'traj': traj_keep, 'cov': cov_keep
        }

        # Mode 1: Change Left (Low Probability, higher uncertainty)
        prob_left = 0.2
        traj_left = np.zeros((N, 2))
        cov_left = np.zeros(N)
        
        for k in range(N):
            # Assume a simple lateral movement of 1 m/s
            traj_left[k] = [x + vx * (k * dt), y + 1.0 * (k * dt)]
            cov_left[k] = 0.2 * k # Higher uncertainty for a lane change
            
        predictions[config.MODE_CHANGE_LEFT] = {
            'prob': prob_left, 'traj': traj_left, 'cov': cov_left
        }

=======
# dynamics.py
import numpy as np
import config

class EgoVehicle:
    def __init__(self, s, e_y, e_psi, v):
        """
        State x_t: [s, e_y, e_psi, v]
        s: arc length (longitudinal progress)
        e_y: lateral error from centerline (y=0)
        e_psi: heading error relative to centerline
        v: velocity
        """
        self.state = np.array([s, e_y, e_psi, v], dtype=float)

    def step(self, action, dt=config.DT):
        """
        action u_t: [a, psi_dot] (acceleration, yaw rate)
        Euler discretization of the Frenet kinematic  model (Eq. 1 in paper).
        Assuming straight road curvature (kappa = 0).
        """
        a, psi_dot = action
        s, e_y, e_psi, v = self.state

        # Derivatives
        s_dot = v * np.cos(e_psi)
        e_y_dot = v * np.sin(e_psi)
        e_psi_dot = psi_dot
        v_dot = a

        # Update state
        self.state[0] += s_dot * dt
        self.state[1] += e_y_dot * dt
        self.state[2] += e_psi_dot * dt
        self.state[3] += v_dot * dt

        # Clip limits
        self.state[3] = np.clip(self.state[3], config.MIN_VELOCITY, config.MAX_VELOCITY)
        
    def get_state(self):
        return self.state.copy()

class TargetVehicle:
    def __init__(self, x, y, vx, tv_id):
        """
        Target Vehicle modeled in Cartesian coordinates.
        State: [X, Y, v_x, v_y]
        """
        self.state = np.array([x, y, vx, 0.0], dtype=float)
        self.id = tv_id

    def step(self, mode, dt=config.DT):
        """
        LTV update based on the designated maneuver mode.
        """
        x, y, vx, vy = self.state

        if mode == config.MODE_KEEP_LANE:
            # Maintain current lane and speed
            vy_new = 0.0
            vx_new = vx 
        elif mode == config.MODE_CHANGE_LEFT:
            # Move laterally to the left
            vy_new = 1.0  # m/s lateral speed
            vx_new = vx
        elif mode == config.MODE_CHANGE_RIGHT:
            vy_new = -1.0
            vx_new = vx
        else:
            vy_new, vx_new = 0.0, vx

        # Simple kinematic update
        self.state[0] += vx_new * dt
        self.state[1] += vy_new * dt
        self.state[2] = vx_new
        self.state[3] = vy_new

    def get_state(self):
        return self.state.copy()
    
    def get_multimodal_predictions(self, N, dt=config.DT):
        """
        Generates open-loop trajectory predictions for the MPC to evaluate.
        Returns a dictionary of modes containing:
        - 'prob': Probability of this mode occurring
        - 'traj': Array of predicted [X, Y] states over horizon N
        - 'cov': Array of spatial uncertainties (variances) over horizon N
        """
        predictions = {}
        x, y, vx, vy = self.state

        # Mode 0: Keep Lane (High Probability)
        prob_keep = 0.8
        traj_keep = np.zeros((N, 2))
        cov_keep = np.zeros(N) # Uncertainty grows over time
        
        for k in range(N):
            traj_keep[k] = [x + vx * (k * dt), y]
            cov_keep[k] = 0.1 * k # Variance grows linearly with time
            
        predictions[config.MODE_KEEP_LANE] = {
            'prob': prob_keep, 'traj': traj_keep, 'cov': cov_keep
        }

        # Mode 1: Change Left (Low Probability, higher uncertainty)
        prob_left = 0.2
        traj_left = np.zeros((N, 2))
        cov_left = np.zeros(N)
        
        for k in range(N):
            # Assume a simple lateral movement of 1 m/s
            traj_left[k] = [x + vx * (k * dt), y + 1.0 * (k * dt)]
            cov_left[k] = 0.2 * k # Higher uncertainty for a lane change
            
        predictions[config.MODE_CHANGE_LEFT] = {
            'prob': prob_left, 'traj': traj_left, 'cov': cov_left
        }

>>>>>>> a0716a10d0730f25f94851fde7115e1102c0813c
        return predictions