<<<<<<< HEAD
# linearization.py
import numpy as np
import config

class EVLinearizer:
    def __init__(self, dt=config.DT):
        self.dt = dt

    def get_AB_matrices(self, x_ref, u_ref):
        """
        Computes the LTV matrices A_k and B_k evaluated along the reference trajectory (Eq 8).
        x_ref: [s, e_y, e_psi, v] at time k
        u_ref: [a, psi_dot] at time k
        """
        s_ref, ey_ref, epsi_ref, v_ref = x_ref
        
        # A matrix: \partial f / \partial x
        A = np.array([
            [1.0, 0.0, -v_ref * np.sin(epsi_ref) * self.dt,  np.cos(epsi_ref) * self.dt],
            [0.0, 1.0,  v_ref * np.cos(epsi_ref) * self.dt,  np.sin(epsi_ref) * self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # B matrix: \partial f / \partial u
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, self.dt],
            [self.dt, 0.0]
        ])
        
        return A, B
        
    def get_batch_matrices(self, x_ref_seq, u_ref_seq):
        """
        Pre-computes A_k and B_k for the entire prediction horizon.
        """
        N = len(u_ref_seq)
        A_seq = np.zeros((N, 4, 4))
        B_seq = np.zeros((N, 4, 2))
        
        for k in range(N):
            A_seq[k], B_seq[k] = self.get_AB_matrices(x_ref_seq[k], u_ref_seq[k])
            
=======
# linearization.py
import numpy as np
import config

class EVLinearizer:
    def __init__(self, dt=config.DT):
        self.dt = dt

    def get_AB_matrices(self, x_ref, u_ref):
        """
        Computes the LTV matrices A_k and B_k evaluated along the reference trajectory (Eq 8).
        x_ref: [s, e_y, e_psi, v] at time k
        u_ref: [a, psi_dot] at time k
        """
        s_ref, ey_ref, epsi_ref, v_ref = x_ref
        
        # A matrix: \partial f / \partial x
        A = np.array([
            [1.0, 0.0, -v_ref * np.sin(epsi_ref) * self.dt,  np.cos(epsi_ref) * self.dt],
            [0.0, 1.0,  v_ref * np.cos(epsi_ref) * self.dt,  np.sin(epsi_ref) * self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # B matrix: \partial f / \partial u
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, self.dt],
            [self.dt, 0.0]
        ])
        
        return A, B
        
    def get_batch_matrices(self, x_ref_seq, u_ref_seq):
        """
        Pre-computes A_k and B_k for the entire prediction horizon.
        """
        N = len(u_ref_seq)
        A_seq = np.zeros((N, 4, 4))
        B_seq = np.zeros((N, 4, 2))
        
        for k in range(N):
            A_seq[k], B_seq[k] = self.get_AB_matrices(x_ref_seq[k], u_ref_seq[k])
            
>>>>>>> a0716a10d0730f25f94851fde7115e1102c0813c
        return A_seq, B_seq