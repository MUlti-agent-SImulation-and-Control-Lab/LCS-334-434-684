import cvxpy as cp
import numpy as np

class PolicyTree:
    def __init__(self, N, nu=2, nx=4, ny=2):
        """
        N: Prediction Horizon
        nu: EV input dimension [a, psi_dot]
        nx: EV state dimension [s, e_y, e_psi, v]
        ny: TV state dimension [X, Y]
        """
        self.N = N
        self.nu = nu
        self.nx = nx
        self.ny = ny

    def create_policy_variables(self, num_modes, branch_step=1):
        """
        Generates the h, M, K policies for all modes and applies Causality 
        and Information Tree constraints.
        branch_step: The future time step where the TV's true mode is revealed.
        """
        policies = []
        constraints = []
        
        for j in range(num_modes):
            # 1. Initialize Variables
            h = cp.Variable(self.N * self.nu, name=f"h_{j}")
            M = cp.Variable((self.N * self.nu, self.N * self.nx), name=f"M_{j}")
            K = cp.Variable((self.N * self.nu, self.N * self.ny), name=f"K_{j}")
            
            # 2. Enforce Causality on M (Strictly Lower Block Triangular)
            # You cannot react to current or future disturbances
            for k in range(self.N):
                for l in range(k, self.N): # If l >= k, force to zero
                    constraints += [
                        M[k*self.nu : (k+1)*self.nu, l*self.nx : (l+1)*self.nx] == 0
                    ]
                    
            # 3. Enforce Causality on K (Block Diagonal)
            # You react exactly to the current TV state deviation at step k
            for k in range(self.N):
                for l in range(self.N):
                    if k != l: # Off-diagonal blocks must be zero
                        constraints += [
                            K[k*self.nu : (k+1)*self.nu, l*self.ny : (l+1)*self.ny] == 0
                        ]
                        
            policies.append({'h': h, 'M': M, 'K': K})
            
        # 4. Enforce The Information Tree (Branching Constraints)
        # If the mode is revealed at 'branch_step', all policies must be identical 
        # BEFORE that step, because the EV doesn't know which future it's in yet.
        if num_modes > 1 and branch_step > 0:
            base_policy = policies[0]
            for j in range(1, num_modes):
                compare_policy = policies[j]
                
                # Constrain h, M, K to be equal up to the branch_step
                branch_idx_u = branch_step * self.nu
                branch_idx_x = branch_step * self.nx
                branch_idx_y = branch_step * self.ny
                
                constraints += [
                    base_policy['h'][:branch_idx_u] == compare_policy['h'][:branch_idx_u],
                    base_policy['M'][:branch_idx_u, :branch_idx_x] == compare_policy['M'][:branch_idx_u, :branch_idx_x],
                    base_policy['K'][:branch_idx_u, :branch_idx_y] == compare_policy['K'][:branch_idx_u, :branch_idx_y]
                ]

        return policies, constraints

    def get_nominal_predictions(self, policies, A_batch, B_batch, x0_dev):
        """
        Calculates the nominal future state sequence (assuming zero future noise).
        U_nom = h
        X_nom = A_batch * x0_dev + B_batch * h
        """
        nominal_trajectories = []
        for policy in policies:
            h = policy['h']
            
            # X_nom will be a flat CVXPY vector of length (N * nx)
            X_nom = A_batch @ x0_dev + B_batch @ h
            nominal_trajectories.append((X_nom, h))
            
        return nominal_trajectories
    