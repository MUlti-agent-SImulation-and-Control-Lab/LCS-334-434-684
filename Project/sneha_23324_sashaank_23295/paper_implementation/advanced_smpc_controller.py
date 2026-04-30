import numpy as np
import cvxpy as cp
import config
from reference import ReferenceTrajectory
from linearization import EVLinearizer
from batch_matrices import compute_batch_matrices
from policy_tree import PolicyTree

class AdvancedSMPC:
    def __init__(self, horizon=20):
        self.N = horizon
        self.dt = config.DT
        
        # Modules from Phases 1, 2, and 3
        self.ref_gen = ReferenceTrajectory(N=self.N, dt=self.dt)
        self.linearizer = EVLinearizer(dt=self.dt)
        self.policy_tree = PolicyTree(N=self.N)
        
        # Cost Weights
        self.Q_ey = 50.0      
        self.Q_v = 10.0       
        self.R_a = 1.0        
        self.R_psidot = 20.0  
        
        # Risk Allocation Parameters (Phase 4)
        self.max_total_risk = 0.10 # Allow a maximum of 10% overall risk
        self.base_risk_mult = 3.0  # Base standard deviation multiplier (very safe)
        self.risk_relaxation = 15.0 # How much a mode's safety buffer shrinks per unit of risk

    def compute_action(self, ego_state, targets_predictions, target_lane, target_vel):
        num_modes = len(targets_predictions[0]) # Assuming TVs share the same modes for simplicity
        
        # 1. Generate Reference Trajectory (Phase 1)
        x_ref_seq, u_ref_seq = self.ref_gen.get_reference(current_s=ego_state[0], 
                                                          target_lane=target_lane, 
                                                          target_vel=target_vel)
        
        # 2. Linearize and Create Batch Matrices (Phases 1 & 2)
        A_seq, B_seq = self.linearizer.get_batch_matrices(x_ref_seq, u_ref_seq)
        A_batch, B_batch = compute_batch_matrices(A_seq, B_seq)
        
        # Deviation of initial state from the reference
        x0_dev = ego_state - x_ref_seq[0]
        
        # 3. Create Parameterized Policies (Phase 3)
        # Assume the TV's true intention is revealed at step k=3
        policies, constraints = self.policy_tree.create_policy_variables(num_modes=num_modes, branch_step=3)
        
        # Get the nominal state predictions (X = A*x0 + B*h) for each mode
        nominal_trajectories = self.policy_tree.get_nominal_predictions(policies, A_batch, B_batch, x0_dev)
        
        # 4. Risk Allocation Variables (Phase 4)
        # eps_j is the risk allocated to mode j
        eps = cp.Variable(num_modes, nonneg=True)
        
        # Extract mode probabilities (assuming TV 0 represents the primary obstacle)
        p = np.array([data['prob'] for data in targets_predictions[0].values()])
        
        # The total risk (sum of probability * risk_j) must be strictly bounded
        constraints += [p @ eps <= self.max_total_risk]
        
        cost = 0.0
        
        # 5. Build Objective and Constraints for each mode
        for j, policy in enumerate(policies):
            X_nom, h = nominal_trajectories[j]
            
            # Stage costs (Weighted by mode probability)
            mode_cost = 0.0
            for k in range(self.N):
                idx_x = k * self.policy_tree.nx
                idx_u = k * self.policy_tree.nu
                
                # Penalize deviations from reference
                mode_cost += self.Q_ey * cp.square(X_nom[idx_x + 1]) # e_y deviation
                mode_cost += self.Q_v * cp.square(X_nom[idx_x + 3])  # velocity deviation
                mode_cost += self.R_a * cp.square(h[idx_u])          # accel magnitude
                mode_cost += self.R_psidot * cp.square(h[idx_u + 1]) # steering magnitude
                
                # State Constraints (Road boundaries)
                # Absolute e_y = reference_e_y + deviation_e_y
                abs_ey = x_ref_seq[k+1, 1] + X_nom[idx_x + 1]
                constraints += [abs_ey <= config.MAX_EY, abs_ey >= config.MIN_EY]
                
                # Input Constraints
                abs_u = u_ref_seq[k] + h[idx_u : idx_u+2]
                constraints += [abs_u[0] >= config.MIN_ACCEL, abs_u[0] <= config.MAX_ACCEL]
                constraints += [abs_u[1] >= -config.MAX_YAW_RATE, abs_u[1] <= config.MAX_YAW_RATE]
                
                # --- PHASE 4: Convex Multimodal Chance Constraints ---
                # We use a linear relaxation to keep the constraint perfectly DCP compliant for CVXPY.
                # As the optimizer assigns more risk (eps[j]) to this mode, the safety multiplier shrinks!
                mode_safety_multiplier = self.base_risk_mult - (self.risk_relaxation * eps[j])
                
                # Ensure the multiplier doesn't become negative (cannot have less than 0 safety buffer)
                constraints += [mode_safety_multiplier >= 0]
                
                # Apply obstacle avoidance
                for tv_preds in targets_predictions:
                    mode_keys = list(tv_preds.keys())
                    tv_data = tv_preds[mode_keys[j]] # Match TV mode to EV policy
                    
                    pred_s = tv_data['traj'][k, 0]
                    pred_ey = tv_data['traj'][k, 1]
                    std_dev = np.sqrt(tv_data['cov'][k])
                    
                    # Absolute EV s position = reference_s + deviation_s
                    abs_s = x_ref_seq[k+1, 0] + X_nom[idx_x]
                    
                    # Dynamic Collision Boundary
                    dynamic_buffer = config.COLLISION_A + (mode_safety_multiplier * std_dev)
                    
                    if abs(pred_ey - target_lane) < (config.LANE_WIDTH / 2) and pred_s > ego_state[0]:
                        constraints += [abs_s <= pred_s - dynamic_buffer]

            # Add the weighted cost of this mode to the total expected cost
            cost += p[j] * mode_cost

        # 6. Solve the Convex Problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            prob.solve(solver=cp.ECOS, warm_start=True)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # We return the very first control action of the baseline policy (h)
                best_h = policies[0]['h'].value
                u_0 = u_ref_seq[0] + best_h[0:2]
                return u_0
            else:
                print(f"Advanced SMPC Failed! Status: {prob.status}")
                return np.array([-3.0, 0.0])
        except Exception as e:
            print(f"Advanced SMPC Error: {e}")
            return np.array([-3.0, 0.0])