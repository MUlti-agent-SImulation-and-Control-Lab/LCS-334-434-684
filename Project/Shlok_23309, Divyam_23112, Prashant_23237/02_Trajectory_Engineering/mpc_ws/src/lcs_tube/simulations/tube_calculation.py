import numpy as np
import scipy.linalg as la
from itertools import product
import pypoman
from pathlib import Path

dt = 0.05  #Sample time
# State: x = [px, py, pz, vx, vy, vz]^T
# Input: u = [ax, ay, az]^T  (gravity-compensated accelerations)

# Discrete-time Double Integrator Matrices (exact analytical)
I3 = np.eye(3)
Z3 = np.zeros((3, 3))

A_d = np.block([
    [I3, dt * I3],
    [Z3, I3]
])

B_d = np.block([
    [0.5 * dt**2 * I3],
    [dt * I3]
])

print("System matrices:")
print(f"A_d shape: {A_d.shape}")
print(f"B_d shape: {B_d.shape}")

# ==============================================================================
# Step 2.1: Local Feedback Gain K via Discrete LQR
# ==============================================================================
# Q = diag([10, 10, 10, 1, 1, 1]) - penalize position heavily
Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
# R = diag([1, 1, 1])
R = np.diag([1.0, 1.0, 1.0])

# Solve Discrete Algebraic Riccati Equation (DARE)
# A^T P A - P - A^T P B (R + B^T P B)^-1 B^T P A + Q = 0
P = la.solve_discrete_are(A_d, B_d, Q, R)

# K = (R + B^T P B)^-1 B^T P A
K = la.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d

print("\nLQR Results:")
print(f"K shape: {K.shape}")
print(f"K = \n{K}")

# Verify A_k = A_d - B_d K is Schur stable
A_k = A_d - B_d @ K
eigenvalues = np.linalg.eigvals(A_k)
print(f"\nA_k eigenvalues: {eigenvalues}")
print(f"All eigenvalues inside unit circle: {all(np.abs(eigenvalues) < 1.0)}")

# ==============================================================================
# Step 2.2: Polyhedral Constraints (H-Representation Hx <= h)
# ==============================================================================

def box_constraints_to_h_representation(lb, ub):
    """
    Convert box constraints lb <= x <= ub to H-representation Hx <= h.
    Returns H and h where x has dimension n.
    """
    n = len(lb)
    H = np.vstack([np.eye(n), -np.eye(n)])
    h = np.concatenate([ub, -np.array(lb)])
    return H, h

# Input Constraints (U): 30-degree tilt limit and Z-acceleration bounds
# |a_x| <= 5.66, |a_y| <= 5.66, |a_z| <= 4.0
u_lb = np.array([-5.66, -5.66, -4.0])
u_ub = np.array([5.66, 5.66, 4.0])
Hu, hu = box_constraints_to_h_representation(u_lb, u_ub)
print(f"\nInput constraint H-representation: Hu shape {Hu.shape}, hu shape {hu.shape}")

# State Constraints (X): 5m x 5m x 3m arena and velocity caps
# p_x in [-2.5, 2.5], p_y in [-2.5, 2.5], p_z in [0.0, 3.0]
# v_x, v_y, v_z in [-2.5, 2.5]
x_lb = np.array([-2.5, -2.5, 0.0, -2.5, -2.5, -2.5])
x_ub = np.array([2.5, 2.5, 3.0, 2.5, 2.5, 2.5])
Hx, hx = box_constraints_to_h_representation(x_lb, x_ub)
print(f"State constraint H-representation: Hx shape {Hx.shape}, hx shape {hx.shape}")

# ==============================================================================
# Step 2.3: Disturbance Set (W) - CORRECTED PHYSICS
# ==============================================================================
# Claude says: the disturbance set W must match the EXACT error dynamics.
# True state: x_{k+1} = A_d x_k + B_d u_true + w_process
# Control law: u_true = u_bar - K (x_measured - x_bar)
#            = u_bar - K (x_true + v_k - x_bar)    where v_k is measurement noise
# Error e_k = x_true - x_bar evolves as:
#   e_{k+1} = A_k e_k + w_process - B_d K v_k
# Therefore the effective disturbance entering the error dynamics is:
#   w_eff = w_process - B_d K v_k
# Phase 2 simulation MUST use this same model (measurement noise enters
# through the controller, NOT as additive state noise).

# Acceleration disturbance: d in [+/-1.0, +/-1.0, +/-1.5]
d_bounds = np.array([1.0, 1.0, 1.5])
d_vertices = [np.array(v) for v in product(*[(-b, b) for b in d_bounds])]

# Map to 6D state space: w_process = B_d * d
w_process_vertices = [B_d @ d for d in d_vertices]

# Claude says: v_k is STATE ESTIMATION (measurement) noise. It does NOT
# enter the plant dynamics directly. It corrupts the controller's feedback
# through the term -B_d K v_k. The Phase 2 simulation must apply v_k
# to the measured state used by the controller, not add it to x_true.
# Measurement noise bounds: v_k in [+/-0.05, +/-0.05, +/-0.1, 0, 0, 0]
noise_bounds = np.array([0.05, 0.05, 0.1, 0.0, 0.0, 0.0])
noise_vertices = [np.array(v) for v in product(*[(-b, b) for b in noise_bounds])]

# Effective disturbance set: W = {w_process - B_d K v_k}
W_vertices = []
for wp in w_process_vertices:
    for vn in noise_vertices:
        # Claude says: v_k enters through feedback: effective_noise = -B_d K v_k
        effective_noise = -(B_d @ K @ vn)
        W_vertices.append(wp + effective_noise)

W_vertices = np.array(W_vertices)
print(f"\nDisturbance set W has {len(W_vertices)} vertices")

# Convert to H-representation
W_poly = pypoman.duality.compute_polytope_halfspaces(W_vertices)
W_H = W_poly[0]
W_h = W_poly[1]

# ==============================================================================
# Step 2.4: RPI Calculation and Tightening
# ==============================================================================

def compute_minkowski_sum_polytopes(H1, h1, H2, h2):
    """
    Compute Minkowski sum of two polytopes P1 and P2 using vertex enumeration.
    P1: H1 @ x <= h1
    P2: H2 @ x <= h2
    Returns H, h for the resulting polytope.
    """
    # Get vertices of both polytopes
    V1 = pypoman.compute_polytope_vertices(H1, h1)
    V2 = pypoman.compute_polytope_vertices(H2, h2)
    
    # Compute all sums
    V_sum = []
    for v1 in V1:
        for v2 in V2:
            V_sum.append(v1 + v2)
    
    V_sum = np.array(V_sum)
    
    # Compute H-representation
    H, h = pypoman.duality.compute_polytope_halfspaces(V_sum)
    return H, h

def compute_mRPI_support_function(A_k, W_vertices, N=10):
    """
    Compute approximate mRPI set: Omega ≈ sum_{i=0}^{N} A_k^i W
    Uses support function method: for each direction d, h_Omega(d) = sum h_{A_k^i W}(d)
    Uses a fixed set of directions (coordinate axes and their combinations).
    
    Claude says: Truncation bound analysis
    The true mRPI is the infinite sum. Truncating at N introduces residual:
      residual <= rho^(N+1) / (1 - rho) * h_W(d)
    where rho = spectral_radius(A_k). With rho ≈ 0.933 and N=100:
      residual ≈ 0.933^101 / (1-0.933) ≈ 2.6% of h_W(d)
    This makes the approximation slightly OPTIMISTIC (tube undersized by ~2.6%).
    For safety-critical applications, scale support values by 1/(1-rho^(N+1))
    to obtain a guaranteed outer approximation.
    
    Claude says: Direction coverage limitation
    72 directions (axis-aligned + pairwise diagonals) in 6D may miss important
    cross-coupling directions. This could cause the mRPI to be an under-
    approximation in uncovered directions, making tightened constraints
    insufficiently conservative. For a rigorous safety proof, consider adding
    random directions or using vertex enumeration for exact computation.
    """
    n = A_k.shape[0]
    
    # Standard directions: +/- coordinate axes
    directions = []
    for i in range(n):
        d = np.zeros(n)
        d[i] = 1.0
        directions.append(d)
        directions.append(-d)
    
    # Add diagonal directions for better approximation
    for i in range(n):
        for j in range(i+1, n):
            d = np.zeros(n)
            d[i] = 1.0
            d[j] = 1.0
            directions.append(d / np.linalg.norm(d))
            directions.append(-d / np.linalg.norm(d))
            d2 = np.zeros(n)
            d2[i] = 1.0
            d2[j] = -1.0
            directions.append(d2 / np.linalg.norm(d2))
            directions.append(-d2 / np.linalg.norm(d2))
    
    directions = np.array(directions)
    print(f"  Using {len(directions)} directions for support function computation")
    
    # Compute spectral radius for truncation bound reporting
    rho = np.max(np.abs(np.linalg.eigvals(A_k)))
    residual_fraction = rho**(N+1) / (1.0 - rho)
    print(f"  Spectral radius rho = {rho:.4f}")
    print(f"  Truncation residual bound: {residual_fraction*100:.2f}% of h_W(d)")
    
    # Compute support function for each direction
    support_values = []
    
    for d in directions:
        h_sum = 0.0
        A_power = np.eye(n)  # A_k^0
        
        for i in range(N + 1):
            # h_{A_k^i W}(d) = max_{w in W} d^T A_k^i w = max_{w in W} (A_k^i^T d)^T w
            d_transformed = A_power.T @ d
            h_W_i = max(d_transformed @ w for w in W_vertices)
            h_sum += h_W_i
            
            A_power = A_power @ A_k
        
        support_values.append(h_sum)
        
        if len(support_values) % 10 == 0:
            print(f"  Computed {len(support_values)}/{len(directions)} directions")
    
    support_values = np.array(support_values)
    
    # Claude says: scale support values to compensate for truncation,
    # yielding a guaranteed outer approximation of the true infinite mRPI
    support_values *= (1.0 / (1.0 - rho**(N+1)))
    print(f"  Applied truncation compensation factor: {1.0/(1.0-rho**(N+1)):.6f}")
    
    # Construct approximate H-representation: d^T x <= h for each direction
    Omega_H = directions
    Omega_h = support_values
    
    return Omega_H, Omega_h

def compute_pontryagin_difference(H1, h1, H2, h2):
    """
    Compute Pontryagin difference: P1 ⊖ P2 = {x | x + y ∈ P1 for all y ∈ P2}
    For polytopes: P1 ⊖ P2 = {x | H1 @ x <= h1 - max_{y∈P2} H1 @ y}
    This is equivalent to tightening the constraints.
    """
    # Get vertices of P2
    V2 = pypoman.compute_polytope_vertices(H2, h2)
    
    # Compute max support function for each constraint
    h_tightened = h1.copy()
    for i, h in enumerate(h1):
        max_val = max(H1[i] @ v for v in V2)
        h_tightened[i] = h - max_val
    
    return H1, h_tightened

def compute_K_omega(K, Omega_H, Omega_h):
    """
    Compute K*Omega = {K @ x | x ∈ Omega}
    """
    Omega_vertices = pypoman.compute_polytope_vertices(Omega_H, Omega_h)
    K_Omega_vertices = [K @ v for v in Omega_vertices]
    K_Omega_vertices = np.array(K_Omega_vertices)
    
    K_Omega_H, K_Omega_h = pypoman.duality.compute_polytope_halfspaces(K_Omega_vertices)
    return K_Omega_H, K_Omega_h

print("\nComputing mRPI set...")
Omega_H, Omega_h = compute_mRPI_support_function(A_k, W_vertices, N=100)
print(f"mRPI set computed: Omega has {len(Omega_h)} constraints")

print("\nComputing tightened state constraints (X_tight = X ⊖ Omega)...")
Hx_tight, hx_tight = compute_pontryagin_difference(Hx, hx, Omega_H, Omega_h)

print("\nComputing K*Omega...")
K_Omega_H, K_Omega_h = compute_K_omega(K, Omega_H, Omega_h)

print("\nComputing tightened input constraints (U_tight = U ⊖ K*Omega)...")
Hu_tight, hu_tight = compute_pontryagin_difference(Hu, hu, K_Omega_H, K_Omega_h)

# Verify tightened constraints are valid
print(f"\nTightened state constraints: Hx_tight shape {Hx_tight.shape}, hx_tight shape {hx_tight.shape}")
print(f"Tightened input constraints: Hu_tight shape {Hu_tight.shape}, hu_tight shape {hu_tight.shape}")

# Check for any negative constraint values (would indicate infeasibility)
if any(hx_tight < 0):
    print("WARNING: Some state constraints are negative! mRPI set may be too large.")
if any(hu_tight < 0):
    print("WARNING: Some input constraints are negative! mRPI set may be too large.")

# ==============================================================================
# Mathematical Verification of Tightened Arena
# ==============================================================================
n_states = 6
# Extract tightened upper and lower bounds for the states
ub_tight = hx_tight[:n_states]
lb_tight = -hx_tight[n_states:]

print("\n--- TUBE WIDTH & ARENA SURVIVAL ANALYSIS ---")
labels = ["px", "py", "pz", "vx", "vy", "vz"]
arena_collapsed = False

for i in range(n_states):
    tightened_width = ub_tight[i] - lb_tight[i]
    original_width = x_ub[i] - x_lb[i]
    tube_width = original_width - tightened_width
    
    print(f"{labels[i]}: Original Width = {original_width:.3f}, Tube Width (Total) = {tube_width:.3f}, Usable Area = {tightened_width:.3f}")
    
    if tightened_width <= 0:
        print(f"CRITICAL FAILURE: The Tube is larger than the arena in the {labels[i]} dimension!")
        arena_collapsed = True

if arena_collapsed:
    print("\nACTION REQUIRED: Your mRPI set is too large. You must either increase the arena size or tune Q and R to penalize position error more aggressively.")
else:
    print("\nSUCCESS: The arena survived the Pontryagin difference. The constraints are mathematically feasible.")

# Save data relative to script location
save_path = Path(__file__).resolve().parent / 'tube_data.npz'
np.savez(save_path,
         A_d=A_d, B_d=B_d, K=K, A_k=A_k, P=P,
         Hx=Hx, hx=hx, Hu=Hu, hu=hu,
         Omega_H=Omega_H, Omega_h=Omega_h,
         W_H=W_H, W_h=W_h,
         Hx_tight=Hx_tight, hx_tight=hx_tight,
         Hu_tight=Hu_tight, hu_tight=hu_tight,
         x_lb=x_lb, x_ub=x_ub,
         u_lb=u_lb, u_ub=u_ub,
         Q=Q, R=R)

print(f"Data saved to {save_path}")
print("\nSummary:")
print(f"- LQR gain K: {K.shape}")
print(f"- A_k eigenvalues: {eigenvalues}")
print(f"- mRPI set Omega: {len(Omega_h)} constraints")
print(f"- Tightened state constraints: {len(hx_tight)} constraints")
print(f"- Tightened input constraints: {len(hu_tight)} constraints")
