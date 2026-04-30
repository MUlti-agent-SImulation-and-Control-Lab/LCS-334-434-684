<<<<<<< HEAD
# mpc_controller.py
import numpy as np
import cvxpy as cp
import config

class DeterministicMPC:
    def __init__(self, horizon=20):
        self.N = horizon  # Prediction horizon (20 steps = 2.0 seconds)
        self.dt = config.DT
        
        # Cost Weights
        self.Q_ey = 50.0      # High penalty for leaving the target lane
        self.Q_v = 10.0       # Moderate penalty for deviating from target speed
        self.Q_epsi = 5.0     # Penalty for pointing away from the lane direction
        self.R_a = 1.0        # Penalty on acceleration (smoothness)
        self.R_psidot = 20.0  # High penalty on steering rate (prevents jerky steering)

    def compute_action(self, ego_state, target_states, target_lane, target_vel):
        """
        Solves the MPC optimization problem.
        ego_state: [s, e_y, e_psi, v]
        target_states: list of [X, Y, vx, vy] for each TV
        """
        # Define Optimization Variables
        # z = [s, e_y, e_psi, v]
        z = cp.Variable((4, self.N + 1))
        # u = [a, psi_dot]
        u = cp.Variable((2, self.N))
        
        cost = 0.0
        constraints = []
        
        # 1. Initial State Constraint
        constraints += [z[:, 0] == ego_state]
        
        for k in range(self.N):
            # 2. Linearized Dynamics Constraints
            # Using target_vel as the v_ref for the e_y linearization
            constraints += [z[0, k+1] == z[0, k] + z[3, k] * self.dt]
            constraints += [z[1, k+1] == z[1, k] + target_vel * z[2, k] * self.dt]
            constraints += [z[2, k+1] == z[2, k] + u[1, k] * self.dt]
            constraints += [z[3, k+1] == z[3, k] + u[0, k] * self.dt]
            
            # 3. Control Limits (Actuator constraints)
            constraints += [u[0, k] >= config.MIN_ACCEL]
            constraints += [u[0, k] <= config.MAX_ACCEL]
            constraints += [u[1, k] >= -config.MAX_YAW_RATE]
            constraints += [u[1, k] <= config.MAX_YAW_RATE]
            
            # 4. State Limits (Road Edges and Speed Limits)
            constraints += [z[3, k+1] >= config.MIN_VELOCITY]
            constraints += [z[3, k+1] <= config.MAX_VELOCITY]
            
            # --- THE ROAD BOUNDARY CONSTRAINTS YOU SUGGESTED ---
            constraints += [z[1, k+1] <= config.MAX_EY]
            constraints += [z[1, k+1] >= config.MIN_EY]
            
            # 5. Stage Cost
            cost += self.Q_ey * cp.square(z[1, k] - target_lane)
            cost += self.Q_v * cp.square(z[3, k] - target_vel)
            cost += self.Q_epsi * cp.square(z[2, k])
            cost += self.R_a * cp.square(u[0, k])
            cost += self.R_psidot * cp.square(u[1, k])
            
            # 6. Collision Avoidance (Deterministic Linear Half-Plane)
            # Check if a target is in our target lane and ahead of us.
            for tv in target_states:
                tv_x, tv_y, tv_vx, tv_vy = tv
                # Predict TV's X position (which maps to 's' on our straight road)
                pred_tv_s = tv_x + tv_vx * (k * self.dt)
                pred_tv_ey = tv_y + tv_vy * (k * self.dt)
                
                # If the TV is laterally close to our target lane, and ahead of us
                if abs(pred_tv_ey - target_lane) < (config.LANE_WIDTH / 2) and pred_tv_s > ego_state[0]:
                    # We must stay strictly behind the TV minus our safety buffer
                    constraints += [z[0, k+1] <= pred_tv_s - config.COLLISION_A]

        # Terminal Cost
        cost += self.Q_ey * cp.square(z[1, self.N] - target_lane)
        cost += self.Q_v * cp.square(z[3, self.N] - target_vel)

        # Solve the QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            # ECOS or OSQP are fast solvers for this
            prob.solve(solver=cp.OSQP, warm_start=True) 
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u[:, 0].value  # Return the optimal [a, psi_dot] for the current step
            else:
                print(f"MPC Failed! Status: {prob.status}. Applying emergency brakes.")
                return np.array([-3.0, 0.0]) # Brake straight
                
        except Exception as e:
            print(f"MPC Error: {e}")
=======
# mpc_controller.py
import numpy as np
import cvxpy as cp
import config

class DeterministicMPC:
    def __init__(self, horizon=20):
        self.N = horizon  # Prediction horizon (20 steps = 2.0 seconds)
        self.dt = config.DT
        
        # Cost Weights
        self.Q_ey = 50.0      # High penalty for leaving the target lane
        self.Q_v = 10.0       # Moderate penalty for deviating from target speed
        self.Q_epsi = 5.0     # Penalty for pointing away from the lane direction
        self.R_a = 1.0        # Penalty on acceleration (smoothness)
        self.R_psidot = 20.0  # High penalty on steering rate (prevents jerky steering)

    def compute_action(self, ego_state, target_states, target_lane, target_vel):
        """
        Solves the MPC optimization problem.
        ego_state: [s, e_y, e_psi, v]
        target_states: list of [X, Y, vx, vy] for each TV
        """
        # Define Optimization Variables
        # z = [s, e_y, e_psi, v]
        z = cp.Variable((4, self.N + 1))
        # u = [a, psi_dot]
        u = cp.Variable((2, self.N))
        
        cost = 0.0
        constraints = []
        
        # 1. Initial State Constraint
        constraints += [z[:, 0] == ego_state]
        
        for k in range(self.N):
            # 2. Linearized Dynamics Constraints
            # Using target_vel as the v_ref for the e_y linearization
            constraints += [z[0, k+1] == z[0, k] + z[3, k] * self.dt]
            constraints += [z[1, k+1] == z[1, k] + target_vel * z[2, k] * self.dt]
            constraints += [z[2, k+1] == z[2, k] + u[1, k] * self.dt]
            constraints += [z[3, k+1] == z[3, k] + u[0, k] * self.dt]
            
            # 3. Control Limits (Actuator constraints)
            constraints += [u[0, k] >= config.MIN_ACCEL]
            constraints += [u[0, k] <= config.MAX_ACCEL]
            constraints += [u[1, k] >= -config.MAX_YAW_RATE]
            constraints += [u[1, k] <= config.MAX_YAW_RATE]
            
            # 4. State Limits (Road Edges and Speed Limits)
            constraints += [z[3, k+1] >= config.MIN_VELOCITY]
            constraints += [z[3, k+1] <= config.MAX_VELOCITY]
            
            # --- THE ROAD BOUNDARY CONSTRAINTS YOU SUGGESTED ---
            constraints += [z[1, k+1] <= config.MAX_EY]
            constraints += [z[1, k+1] >= config.MIN_EY]
            
            # 5. Stage Cost
            cost += self.Q_ey * cp.square(z[1, k] - target_lane)
            cost += self.Q_v * cp.square(z[3, k] - target_vel)
            cost += self.Q_epsi * cp.square(z[2, k])
            cost += self.R_a * cp.square(u[0, k])
            cost += self.R_psidot * cp.square(u[1, k])
            
            # 6. Collision Avoidance (Deterministic Linear Half-Plane)
            # Check if a target is in our target lane and ahead of us.
            for tv in target_states:
                tv_x, tv_y, tv_vx, tv_vy = tv
                # Predict TV's X position (which maps to 's' on our straight road)
                pred_tv_s = tv_x + tv_vx * (k * self.dt)
                pred_tv_ey = tv_y + tv_vy * (k * self.dt)
                
                # If the TV is laterally close to our target lane, and ahead of us
                if abs(pred_tv_ey - target_lane) < (config.LANE_WIDTH / 2) and pred_tv_s > ego_state[0]:
                    # We must stay strictly behind the TV minus our safety buffer
                    constraints += [z[0, k+1] <= pred_tv_s - config.COLLISION_A]

        # Terminal Cost
        cost += self.Q_ey * cp.square(z[1, self.N] - target_lane)
        cost += self.Q_v * cp.square(z[3, self.N] - target_vel)

        # Solve the QP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            # ECOS or OSQP are fast solvers for this
            prob.solve(solver=cp.OSQP, warm_start=True) 
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return u[:, 0].value  # Return the optimal [a, psi_dot] for the current step
            else:
                print(f"MPC Failed! Status: {prob.status}. Applying emergency brakes.")
                return np.array([-3.0, 0.0]) # Brake straight
                
        except Exception as e:
            print(f"MPC Error: {e}")
>>>>>>> a0716a10d0730f25f94851fde7115e1102c0813c
            return np.array([-3.0, 0.0])