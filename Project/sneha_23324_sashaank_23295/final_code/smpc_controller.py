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
        True Tree-Structured Stochastic MPC
        """
        ego_s = current_state[0]
        
        # 1. Identify the Primary Threat (Closest TV ahead)
        primary_gmm = None
        min_s_diff = 100.0
        
        for tv_gmm in tv_predictions:
            # Check the first mode to find distance
            s_diff = tv_gmm[0]['trajectory'][0, 0] - ego_s
            if s_diff < -config.TRACK_LENGTH/2: s_diff += config.TRACK_LENGTH
            
            if 0 < s_diff < min_s_diff:
                min_s_diff = s_diff
                primary_gmm = tv_gmm
                
        # If the road is clear, create a single "dummy" mode to keep math running
        if primary_gmm is None or min_s_diff > 40.0:
            primary_gmm = [{'weight': 1.0, 'trajectory': np.zeros((self.N, 4))}]
            min_s_diff = 999.0

        num_modes = len(primary_gmm)
        
        # --- BUILD TREE-STRUCTURED MPC PROBLEM ---
        # We now create separate Variables for EVERY branch in the tree
        xs = [cp.Variable((self.nx, self.N + 1)) for _ in range(num_modes)]
        us = [cp.Variable((self.nu, self.N)) for _ in range(num_modes)]
        
        A, B = self.get_linearized_dynamics(config.V_REF)
        
        # Our target state is simply to cruise in the center lane
        x_ref = np.array([0.0, 0.0, 0.0, config.V_REF])
        
        cost = 0.0
        constraints = []
        
        # Build the branches
        for m, mode in enumerate(primary_gmm):
            weight = mode['weight']
            tv_traj = mode['trajectory']
            
            # Initial state constraint (All branches start from the same physical reality)
            constraints += [xs[m][:, 0] == current_state]
            
            # --- BRANCH-SPECIFIC SAFE TARGET ---
            x_ref_branch = np.array([0.0, 0.0, 0.0, config.V_REF])
            current_lane_ey = round(current_state[1] / config.LANE_WIDTH) * config.LANE_WIDTH
            
            if min_s_diff <= 40.0:
                blocked_lanes = set()
                
                # 1. Block the lane used by THIS specific branch of the primary threat
                tv_ey = tv_traj[-1, 1] 
                primary_threat_lane = round(tv_ey / config.LANE_WIDTH) * config.LANE_WIDTH
                blocked_lanes.add(primary_threat_lane)
                
                # 2. Block lanes occupied by ALL OTHER vehicles within 40m
                for other_gmm in tv_predictions:
                    if other_gmm is primary_gmm: continue # Already handled above
                    other_traj = other_gmm[0]['trajectory']
                    s_diff = other_traj[0, 0] - current_state[0]
                    if s_diff < -config.TRACK_LENGTH/2: s_diff += config.TRACK_LENGTH
                    
                    if 0 < s_diff < 40.0:
                        other_lane = round(other_traj[0, 1] / config.LANE_WIDTH) * config.LANE_WIDTH
                        blocked_lanes.add(other_lane)

                # 3. Choose the safest route, or BRAKE!
                possible_lanes = [0.0, config.LANE_WIDTH, -config.LANE_WIDTH]
                safe_lanes = [lane for lane in possible_lanes if lane not in blocked_lanes]
                
                if safe_lanes:
                    # Pick the safe lane closest to where we currently are
                    x_ref_branch[1] = min(safe_lanes, key=lambda x: abs(x - current_lane_ey))
                else:
                    # SCENARIO: ALL LANES BLOCKED!
                    x_ref_branch[1] = current_lane_ey # Keep the wheel straight
                    x_ref_branch[3] = 0.0             # Target Velocity = 0 (SLAM BRAKES!)

            for k in range(self.N):
                # EXPECTED COST: Use the branch-specific safe target!
                state_error = xs[m][:, k] - x_ref_branch
                cost += weight * cp.quad_form(state_error, self.Q)
                cost += weight * cp.quad_form(us[m][:, k], self.R)
                
                # Dynamics and Physical Limits for this branch
                constraints += [xs[m][:, k+1] == A @ xs[m][:, k] + B @ us[m][:, k]]
                constraints += [us[m][0, k] <= config.MAX_ACCEL]
                constraints += [us[m][0, k] >= config.MIN_ACCEL]
                constraints += [us[m][1, k] <= config.MAX_YAW_RATE]
                constraints += [us[m][1, k] >= -config.MAX_YAW_RATE]
                
                # Track Boundaries (Keep vehicle on the road)
                constraints += [xs[m][1, k] <= 1.5 * config.LANE_WIDTH]
                constraints += [xs[m][1, k] >= -1.5 * config.LANE_WIDTH]

            # Terminal Expected Cost
            cost += weight * cp.quad_form(xs[m][:, self.N] - x_ref_branch, self.Q)
            
        # --- NON-ANTICIPATIVITY CONSTRAINTS ---
        # The Ego car MUST make the exact same initial steering/acceleration decision, 
        # regardless of which mode the TV takes, because we don't know the future yet!
        for m in range(1, num_modes):
            constraints += [us[m][:, 0] == us[0][:, 0]]
            
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # Solve the massive multi-branch optimization problem
        prob.solve(solver=cp.OSQP, warm_start=True) 
        
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # We execute the common "trunk" of the tree
            return us[0].value[0, 0], us[0].value[1, 0]
        else:
            print("SMPC Mathematics Infeasible! Emergency Braking!")
            return -config.MAX_ACCEL, 0.0 # Slam brakes, no steering
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


