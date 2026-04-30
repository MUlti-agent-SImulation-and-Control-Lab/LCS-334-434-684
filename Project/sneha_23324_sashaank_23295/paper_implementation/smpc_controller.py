# smpc_controller.py
import numpy as np
import cvxpy as cp
import config

class StochasticMPC:
    def __init__(self, horizon=20):
        self.N = horizon
        self.dt = config.DT
        
        # Cost Weights (Same as before)
        self.Q_ey = 50.0      
        self.Q_v = 10.0       
        self.Q_epsi = 5.0     
        self.R_a = 1.0        
        self.R_psidot = 20.0  
        
        # SMPC Parameters
        self.prob_threshold = 0.1 # Ignore modes with <10% probability
        self.risk_factor = 2.0    # Multiplier for the standard deviation (approx 95% confidence)

    def compute_action(self, ego_state, targets_predictions, target_lane, target_vel):
        z = cp.Variable((4, self.N + 1))
        u = cp.Variable((2, self.N))
        cost = 0.0
        constraints = []
        
        # Initial State
        constraints += [z[:, 0] == ego_state]
        
        for k in range(self.N):
            # Linearized Dynamics
            constraints += [z[0, k+1] == z[0, k] + z[3, k] * self.dt]
            constraints += [z[1, k+1] == z[1, k] + target_vel * z[2, k] * self.dt]
            constraints += [z[2, k+1] == z[2, k] + u[1, k] * self.dt]
            constraints += [z[3, k+1] == z[3, k] + u[0, k] * self.dt]
            
            # Control & State Limits
            constraints += [u[0, k] >= config.MIN_ACCEL, u[0, k] <= config.MAX_ACCEL]
            constraints += [u[1, k] >= -config.MAX_YAW_RATE, u[1, k] <= config.MAX_YAW_RATE]
            constraints += [z[3, k+1] >= config.MIN_VELOCITY, z[3, k+1] <= config.MAX_VELOCITY]
            constraints += [z[1, k+1] <= config.MAX_EY, z[1, k+1] >= config.MIN_EY]
            
            # Stage Cost
            cost += self.Q_ey * cp.square(z[1, k] - target_lane)
            cost += self.Q_v * cp.square(z[3, k] - target_vel)
            cost += self.Q_epsi * cp.square(z[2, k])
            cost += self.R_a * cp.square(u[0, k])
            cost += self.R_psidot * cp.square(u[1, k])
            
            # --- CHANCE CONSTRAINTS (MULTIMODAL) ---
            for tv_preds in targets_predictions:
                for mode, data in tv_preds.items():
                    if data['prob'] >= self.prob_threshold:
                        pred_s = data['traj'][k, 0]
                        pred_ey = data['traj'][k, 1]
                        std_dev = np.sqrt(data['cov'][k])
                        
                        # Dynamic Safety Buffer: Base Ellipse + (Risk Factor * Uncertainty)
                        dynamic_buffer = config.COLLISION_A + (self.risk_factor * std_dev)
                        
                        # If TV mode is near our target lane and ahead of us
                        if abs(pred_ey - target_lane) < (config.LANE_WIDTH / 2) and pred_s > ego_state[0]:
                            constraints += [z[0, k+1] <= pred_s - dynamic_buffer]

        cost += self.Q_ey * cp.square(z[1, self.N] - target_lane)
        cost += self.Q_v * cp.square(z[3, self.N] - target_vel)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u[:, 0].value 
            else:
                return np.array([-3.0, 0.0])
        except Exception:
            return np.array([-3.0, 0.0])