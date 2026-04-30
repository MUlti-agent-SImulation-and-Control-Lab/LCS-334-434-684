# Let us create the skeleton for our optimization controller.
#  For this first iteration, we will implement the Nominal MPC. 
# This means we will program it to perfectly track the lane and speed while respecting its physical limits,
#  but we will ignore the other vehicles just for a moment to ensure the optimization loop runs without crashing.



import numpy as np
import cvxpy as cp
import config

class SMPCController:
    def __init__(self):
        self.N = config.N_HORIZON
        self.dt = config.DT
        
        # State and Input Dimensions
        self.nx = 4 # [s, ey, epsi, v]
        self.nu = 2 # [a, yaw_rate]
        
        # Cost Matrices
        self.Q = np.diag([config.Q_S, config.Q_EY, config.Q_EPSI, config.Q_V])
        self.R = np.diag([config.R_A, config.R_YAW])
        
    def get_linearized_dynamics(self, v_ref):
        """
        Returns the discrete-time A and B matrices linearized 
        around the reference speed on a straight road (kappa = 0).
        """
        A = np.eye(self.nx)
        A[0, 3] = self.dt  # s_next = s + v * dt
        A[1, 2] = v_ref * self.dt # ey_next = ey + epsi * v * dt
        
        B = np.zeros((self.nx, self.nu))
        B[2, 1] = self.dt  # epsi_next = epsi + yaw_rate * dt
        B[3, 0] = self.dt  # v_next = v + a * dt
        
        return A, B
    
    # We are going to make our SMPC "aware" of the other vehicles.
    #  We will pass the predictions into the solve function.
    # If the controller detects a slow vehicle ahead in its lane, 
    # it will dynamically change its x_ref (reference target) to the left lane to initiate an overtake.
    def solve(self, current_state, tv_predictions):
        """
        current_state: [s, ey, epsi, v]
        tv_predictions: list of predictions for each TV
        """
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))
        
        A, B = self.get_linearized_dynamics(config.V_REF)
        
        cost = 0.0
        constraints = [x[:, 0] == current_state]
        
        # --- BEHAVIORAL PLANNER (Dynamic Overtake Decision) ---
        # Get current position from state vector
        ego_s = current_state[0]
        ego_ey = current_state[1]
        
        # 1. Figure out which lane we are currently closest to
        current_lane_ey = round(ego_ey / config.LANE_WIDTH) * config.LANE_WIDTH
        
        # 2. Default goal: stay in the lane we are currently in
        target_ey = current_lane_ey 
        target_v = config.V_REF
        
        for tv_pred in tv_predictions:
            tv_s_initial = tv_pred[0, 0]
            tv_ey_initial = tv_pred[0, 1]
            tv_v = tv_pred[0, 3]
            
            # Distance to TV
            s_diff = tv_s_initial - ego_s
            if s_diff < -config.TRACK_LENGTH/2: s_diff += config.TRACK_LENGTH
            
            # If TV is ahead of us, within 40 meters, and in OUR current lane
            if 0 < s_diff < 40.0 and abs(tv_ey_initial - current_lane_ey) < 2.0:
                
                # We need to change lanes! Where do we go?
                if current_lane_ey == 0.0:
                    # If in center, overtake on the left
                    target_ey = config.LANE_WIDTH 
                    print(f"Obstacle in center! Shifting left.")
                elif current_lane_ey == config.LANE_WIDTH:
                    # If in left lane, dodge back to the center
                    target_ey = 0.0 
                    print(f"Obstacle in left lane! Shifting to center.")
                elif current_lane_ey == -config.LANE_WIDTH:
                    # If in right lane, dodge to the center
                    target_ey = 0.0
                    print(f"Obstacle in right lane! Shifting to center.")
                break # React to the first immediate threat in our lane

        # Our new reference state for this time step
        x_ref = np.array([0.0, target_ey, 0.0, target_v])
        
        # --- BUILD MPC PROBLEM ---
        for k in range(self.N):
            state_error = x[:, k] - x_ref
            cost += cp.quad_form(state_error, self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
            constraints += [u[0, k] <= config.MAX_ACCEL]
            constraints += [u[0, k] >= config.MIN_ACCEL]
            constraints += [u[1, k] <= config.MAX_YAW_RATE]
            constraints += [u[1, k] >= -config.MAX_YAW_RATE]
            
            # Track Boundary Constraints (Keep vehicle on the road)
            constraints += [x[1, k] <= 1.5 * config.LANE_WIDTH]  # Left boundary
            constraints += [x[1, k] >= -1.5 * config.LANE_WIDTH] # Right boundary
            
        cost += cp.quad_form(x[:, self.N] - x_ref, self.Q)
        
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True) 
        
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            return u.value[0, 0], u.value[1, 0]
        else:
            return 0.0, 0.0
"""
    def solve(self, current_state):
        
        # Sets up and solves the Convex Optimization problem using CVXPY.
        # current_state: [s, ey, epsi, v]
       
        # Define Optimization Variables
        x = cp.Variable((self.nx, self.N + 1))
        u = cp.Variable((self.nu, self.N))
        
        # Get LTI matrices
        A, B = self.get_linearized_dynamics(config.V_REF)
        
        cost = 0.0
        constraints = []
        
        # Initial State Constraint
        constraints += [x[:, 0] == current_state]
        
        # Reference State [s is ignored by Q, ey=0, epsi=0, v=V_REF]
        x_ref = np.array([0.0, 0.0, 0.0, config.V_REF])
        
        # Build the Cost and Dynamics Constraints over the horizon
        for k in range(self.N):
            # Objective: Minimize tracking error and control effort
            state_error = x[:, k] - x_ref
            cost += cp.quad_form(state_error, self.Q)
            cost += cp.quad_form(u[:, k], self.R)
            
            # Constraint: System Dynamics (x_k+1 = Ax_k + Bu_k)
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
            
            # Constraint: Actuation Limits
            constraints += [u[0, k] <= config.MAX_ACCEL]
            constraints += [u[0, k] >= config.MIN_ACCEL]
            constraints += [u[1, k] <= config.MAX_YAW_RATE]
            constraints += [u[1, k] >= -config.MAX_YAW_RATE]
            
        # Terminal Cost (to ensure stability)
        cost += cp.quad_form(x[:, self.N] - x_ref, self.Q)
        
        # Setup and Solve the Problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # OSQP is a fast, free solver built into CVXPY
        prob.solve(solver=cp.OSQP, warm_start=True) 
        
        if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
            # Return the very first control action of the optimized sequence
            optimal_a = u.value[0, 0]
            optimal_yaw_rate = u.value[1, 0]
            return optimal_a, optimal_yaw_rate
        else:
            print(f"Solver failed! Status: {prob.status}")
            return 0.0, 0.0 # Emergency fallback

"""


