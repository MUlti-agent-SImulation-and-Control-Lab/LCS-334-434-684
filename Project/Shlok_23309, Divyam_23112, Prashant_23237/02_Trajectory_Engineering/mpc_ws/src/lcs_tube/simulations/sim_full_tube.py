import numpy as np
import cvxpy as cp
import scipy.linalg as la

def run_simulation(T_sim=200, seed=None, disturbance_func=None, waypoints=None):
    if seed is not None:
        np.random.seed(seed)
        
    data = np.load('tube_data_gemini.npz', allow_pickle=True)
    A_d = data['A_d']
    B_d = data['B_d']
    K = data['K']
    Hx_tight = data['Hx_tight']
    hx_tight = data['hx_tight']
    Hu_tight = data['Hu_tight']
    hu_tight = data['hu_tight']
    Omega_H = data['Omega_H']
    Omega_h = data["Omega_h"]
    u_lb = data['u_lb']
    u_ub = data['u_ub']

    N = 20
    dt = 0.05
    Q_mpc = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    R_mpc = np.diag([0.1, 0.1, 0.1])
    
    if waypoints is None:
        waypoints = [np.array([0.5, 1.0, 1.5, 0.0, 0.0, 0.0])]
    else:
        waypoints = [np.array(wp) for wp in waypoints]
        
    current_wp_idx = 0
    x_ref = waypoints[current_wp_idx]

    # Claude says: terminal cost uses DARE solution for proper stability certificate
    P_terminal = la.solve_discrete_are(A_d, B_d, Q_mpc, R_mpc)

    X_bar = cp.Variable((6, N + 1))
    U_bar = cp.Variable((3, N))
    x_curr_param = cp.Parameter(6)
    x_ref_param = cp.Parameter(6)

    cost = 0
    for k in range(N):
        cost += cp.quad_form(X_bar[:, k] - x_ref_param, Q_mpc)
        cost += cp.quad_form(U_bar[:, k], R_mpc)
    # Claude says: terminal cost uses P_terminal (DARE solution) instead of Q_mpc
    cost += cp.quad_form(X_bar[:, N] - x_ref_param, P_terminal)

    constraints = []
    constraints.append(Omega_H @ (x_curr_param - X_bar[:, 0]) <= Omega_h)
    for k in range(N):
        constraints.append(X_bar[:, k + 1] == A_d @ X_bar[:, k] + B_d @ U_bar[:, k])
    for k in range(N + 1):
        constraints.append(Hx_tight @ X_bar[:, k] <= hx_tight)
    for k in range(N):
        constraints.append(Hu_tight @ U_bar[:, k] <= hu_tight)

    problem = cp.Problem(cp.Minimize(cost), constraints)

    # Claude says: disturbance model correctly separates process noise and
    # measurement noise, matching Phase 1's error dynamics model exactly.
    def default_disturbance(k):
        """Returns (w_process, v_k) tuple matching Phase 1 error dynamics model."""
        d = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), np.random.uniform(-1.5, 1.5)])
        w_process = B_d @ d
        v_k = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), np.random.uniform(-0.1, 0.1), 0.0, 0.0, 0.0])
        return w_process, v_k
        
    if disturbance_func is None:
        disturbance_func = default_disturbance

    def solve_mpc(x_current, prev_X, prev_U):
        x_curr_param.value = x_current
        x_ref_param.value = x_ref
        X_bar.value = prev_X
        U_bar.value = prev_U
        
        problem.solve(solver=cp.OSQP, warm_start=True)
        if problem.status not in ["optimal", "optimal_inaccurate"] or X_bar.value is None:
            X_fallback = np.roll(prev_X, -1, axis=1)
            U_fallback = np.roll(prev_U, -1, axis=1)
            u_pad = -K @ (X_fallback[:, -2] - x_ref)
            U_fallback[:, -1] = u_pad
            X_fallback[:, -1] = A_d @ X_fallback[:, -2] + B_d @ u_pad
            return X_fallback, U_fallback
        return X_bar.value.copy(), U_bar.value.copy()

    x_true = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    X_nom_traj = np.tile(x_true, (N + 1, 1)).T
    U_nom_traj = np.zeros((3, N))

    trigger_events = []
    clip_violations = 0

    x_true_history = np.zeros((6, T_sim + 1))
    x_nom_history = np.zeros((6, T_sim + 1))
    u_true_history = np.zeros((3, T_sim))
    x_ref_history = np.zeros((6, T_sim + 1))

    x_true_history[:, 0] = x_true
    x_nom_history[:, 0] = x_true
    x_ref_history[:, 0] = x_ref

    X_nom_traj, U_nom_traj = solve_mpc(x_true, X_nom_traj, U_nom_traj)

    for k in range(T_sim):
        # Check waypoint switching
        dist_to_wp = np.linalg.norm(x_true[0:3] - x_ref[0:3])
        if dist_to_wp < 0.2 and current_wp_idx < len(waypoints) - 1:
            current_wp_idx += 1
            x_ref = waypoints[current_wp_idx]
            
        # Full Tube MPC solves at every step
        trigger_events.append((k, x_true[0:3].copy()))
        X_nom_traj, U_nom_traj = solve_mpc(x_true, X_nom_traj, U_nom_traj)
            
        u_bar_k = U_nom_traj[:, 0]
        x_bar_k = X_nom_traj[:, 0]
        
        # Claude says: measurement noise v_k corrupts the controller's state estimate
        w_process, v_k = disturbance_func(k)
        x_measured = x_true + v_k
        u_true = u_bar_k - K @ (x_measured - x_bar_k)
        
        # Claude says: log clip violations as they indicate tube invariance breach
        u_unclipped = u_true.copy()
        u_true = np.clip(u_true, u_lb, u_ub)
        if not np.allclose(u_true, u_unclipped):
            clip_violations += 1
        
        # Claude says: only process noise enters the plant dynamics
        x_true = A_d @ x_true + B_d @ u_true + w_process
        
        x_true_history[:, k + 1] = x_true
        x_nom_history[:, k + 1] = X_nom_traj[:, 0]
        u_true_history[:, k] = u_true
        x_ref_history[:, k + 1] = x_ref
        
        X_nom_traj = np.roll(X_nom_traj, -1, axis=1)
        U_nom_traj = np.roll(U_nom_traj, -1, axis=1)
        
        x_end = X_nom_traj[:, -2]
        u_pad = -K @ (x_end - x_ref)
        U_nom_traj[:, -1] = u_pad
        X_nom_traj[:, -1] = A_d @ x_end + B_d @ u_pad

    if clip_violations > 0:
        print(f"  [!] WARNING: Input clipping was active {clip_violations}/{T_sim} steps — tube invariance violated!")

    return {
        'name': 'Full Tube MPC',
        'x_true_history': x_true_history,
        'x_nom_history': x_nom_history,
        'u_true_history': u_true_history,
        'trigger_events': trigger_events,
        'clip_violations': clip_violations,
        'x_ref_history': x_ref_history,
        'waypoints': waypoints,
        'dt': dt,
        'T_sim': T_sim
    }

if __name__ == "__main__":
    res = run_simulation(200, 42)
    print(f"Full Tube MPC Triggers: {len(res['trigger_events'])}")
