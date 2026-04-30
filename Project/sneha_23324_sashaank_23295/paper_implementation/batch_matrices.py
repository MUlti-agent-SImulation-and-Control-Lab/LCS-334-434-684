import numpy as np

def compute_batch_matrices(A_seq, B_seq):
    """
    Computes the lifted block matrices for the LTV system.
    A_seq: Array of A_k matrices of shape (N, nx, nx)
    B_seq: Array of B_k matrices of shape (N, nx, nu)
    
    Returns:
    A_batch: shape (N*nx, nx)
    B_batch: shape (N*nx, N*nu)
    """
    N, nx, _ = A_seq.shape
    _, _, nu = B_seq.shape
    
    A_batch = np.zeros((N * nx, nx))
    B_batch = np.zeros((N * nx, N * nu))
    
    # We will build the transition matrix Phi(k, j) = A_{k-1} * ... * A_j
    # where Phi(k, k) = I
    
    for k in range(1, N + 1):
        row_start = (k - 1) * nx
        row_end = k * nx
        
        # 1. Compute A_batch row block
        # A_batch[k] = A_{k-1} * A_{k-2} * ... * A_0
        temp_A = np.eye(nx)
        for i in range(k - 1, -1, -1):
            temp_A = A_seq[i] @ temp_A
        A_batch[row_start:row_end, :] = temp_A
        
        # 2. Compute B_batch row blocks (Lower Triangular)
        for j in range(k):
            col_start = j * nu
            col_end = (j + 1) * nu
            
            # B_batch[k, j] = Phi(k, j+1) * B_j
            temp_B = np.eye(nx)
            for i in range(k - 1, j, -1):
                temp_B = A_seq[i] @ temp_B
                
            B_batch[row_start:row_end, col_start:col_end] = temp_B @ B_seq[j]
            
    return A_batch, B_batch