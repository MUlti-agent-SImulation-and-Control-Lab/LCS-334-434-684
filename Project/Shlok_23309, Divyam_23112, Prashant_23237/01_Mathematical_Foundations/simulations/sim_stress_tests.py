"""
Stress Testing Suite for Event-Triggered Tube MPC
Subjects the controller to four edge cases and compares
ET Tube MPC (robust) vs ET MPC No Aux (baseline).
"""

import numpy as np
import cvxpy as cp
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Load offline data
base_dir = Path(__file__).parent
data = np.load(base_dir / 'tube_data.npz', allow_pickle=True)
#data = np.load('tube_data.npz', allow_pickle=True) #won't work with different path
A_d = data['A_d']
B_d = data['B_d']
K   = data['K']
Hx_tight = data['Hx_tight'];  hx_tight = data['hx_tight']
Hu_tight = data['Hu_tight'];  hu_tight = data['hu_tight']
Hx = data['Hx'];  hx = data['hx']
Hu = data['Hu'];  hu = data['hu']
Omega_H = data['Omega_H'];  Omega_h = data['Omega_h']
x_lb = data['x_lb'];  x_ub = data['x_ub']
u_lb = data['u_lb'];  u_ub = data['u_ub']

# Shared constants
N  = 20
dt = 0.05
T_SIM = 200           # 10 seconds
Q_MPC = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
R_MPC = np.diag([0.1, 0.1, 0.1])
P_TERMINAL = la.solve_discrete_are(A_d, B_d, Q_MPC, R_MPC)
P_TERMINAL = (P_TERMINAL + P_TERMINAL.T) / 2.0
WAYPOINTS = [
    np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0]),
    np.array([-0.5, 1.0, 1.5, 0.0, 0.0, 0.0]),
    np.array([0.5, -0.5, 2.0, 0.0, 0.0, 0.0])
]

# Scenario configurations
SCENARIOS = {
    'A_wind': {
        'label': 'A: Persistent Crosswind',
        'x0': np.array([-0.95, 0.0, 0.5, 0.0, 0.0, 0.0]),
        'w_wind': lambda k, x: B_d @ np.array([1.0, 0.0, 0.0]),
        'thrust_scale': 1.0,
        'sensor_glitch_step': None,
    },
    'B_payload': {
        'label': 'B: Payload Drop (100%)',
        'x0': np.array([-0.95, 0.0, 0.5, 0.0, 0.0, 0.0]),
        'w_wind': lambda k, x: np.zeros(6),
        'thrust_scale': 0.35,
        'sensor_glitch_step': None,
    },
    'C_wall': {
        'label': 'C: High-Speed Catch + Gust',
        'x0': np.array([-0.95, 0.0, 0.5, 0.0, 0.0, 0.0]),
        'w_wind': lambda k, x: B_d @ np.array([1.0, 0.0, 0.0]) if x[0] > 0.2 and k < 80 else np.zeros(6),
        'thrust_scale': 1.0,
        'sensor_glitch_step': None,
    },
    'D_glitch': {
        'label': 'D: UWB Sensor Glitch',
        'x0': np.array([-0.95, 0.0, 0.5, 0.0, 0.0, 0.0]),
        'w_wind': lambda k, x: np.zeros(6),
        'thrust_scale': 1.0,
        'sensor_glitch_step': int(1.5 / dt),
    },
}


# ==============================================================================
# Build CVXPY problems once (parametric – reused across all runs)
# ==============================================================================
def _build_problem(controller_type):
    """Build and return (problem, X_bar, U_bar, x_curr_param, x_ref_param)."""
    X_bar = cp.Variable((6, N + 1))
    U_bar = cp.Variable((3, N))
    x_curr = cp.Parameter(6)
    x_ref  = cp.Parameter(6)

    cost = 0
    for k in range(N):
        cost += cp.quad_form(X_bar[:, k] - x_ref, Q_MPC)
        cost += cp.quad_form(U_bar[:, k], R_MPC)
    cost += cp.quad_form(X_bar[:, N] - x_ref, P_TERMINAL)

    cons = []
    if controller_type == 'et_tube':
        cons.append(Omega_H @ (x_curr - X_bar[:, 0]) <= Omega_h)
        for k in range(N):
            cons.append(X_bar[:, k+1] == A_d @ X_bar[:, k] + B_d @ U_bar[:, k])
        for k in range(N + 1):
            cons.append(Hx_tight @ X_bar[:, k] <= hx_tight)
        for k in range(N):
            cons.append(Hu_tight @ U_bar[:, k] <= hu_tight)
    else:
        cons.append(X_bar[:, 0] == x_curr)
        for k in range(N):
            cons.append(X_bar[:, k+1] == A_d @ X_bar[:, k] + B_d @ U_bar[:, k])
        for k in range(N + 1):
            cons.append(Hx @ X_bar[:, k] <= hx)
        for k in range(N):
            cons.append(Hu @ U_bar[:, k] <= hu)

    prob = cp.Problem(cp.Minimize(cost), cons)
    return prob, X_bar, U_bar, x_curr, x_ref

# Pre-build problem templates
prob_tube, Xb_tube, Ub_tube, xp_tube, rp_tube = _build_problem('et_tube')
prob_full, Xb_full, Ub_full, xp_full, rp_full = _build_problem('et_tube')
prob_standard, Xb_standard, Ub_standard, xp_standard, rp_standard = _build_problem('standard')


def run_simulation(scenario_name, controller_type, seed=43):
    """
    Run one simulation.

    Parameters
    ----------
    scenario_name : str   – key into SCENARIOS dict
    controller_type: str  – 'et_tube', 'full_tube', 'no_aux', or 'standard'
    seed          : int

    Returns
    -------
    dict with trajectory history and metrics
    """
    np.random.seed(seed)
    cfg = SCENARIOS[scenario_name]

    # Pick the correct CVXPY objects
    if controller_type == 'et_tube':
        prob, X_var, U_var, xp, rp = prob_tube, Xb_tube, Ub_tube, xp_tube, rp_tube
        ctrl_name = 'ET Tube MPC'
        use_aux_loop = True
    elif controller_type == 'full_tube':
        prob, X_var, U_var, xp, rp = prob_full, Xb_full, Ub_full, xp_full, rp_full
        ctrl_name = 'Full Tube MPC'
        use_aux_loop = True
    elif controller_type == 'standard':
        prob, X_var, U_var, xp, rp = prob_standard, Xb_standard, Ub_standard, xp_standard, rp_standard
        ctrl_name = 'Standard MPC'
        use_aux_loop = False

    def solve_mpc(x_current, prev_X, prev_U, current_x_ref):
        xp.value = x_current
        rp.value = current_x_ref
        X_var.value = prev_X
        U_var.value = prev_U
        prob.solve(solver=cp.OSQP, warm_start=True)
        if prob.status not in ['optimal', 'optimal_inaccurate'] or X_var.value is None:
            # Safety fallback
            Xf = np.roll(prev_X, -1, axis=1)
            Uf = np.roll(prev_U, -1, axis=1)
            if use_aux_loop:
                u_pad = -K @ (Xf[:, -2] - current_x_ref)
            else:
                u_pad = np.zeros(3)
            Uf[:, -1] = u_pad
            Xf[:, -1] = A_d @ Xf[:, -2] + B_d @ u_pad
            return Xf, Uf, False
        return X_var.value.copy(), U_var.value.copy(), True

    # State initialisation
    x_true = cfg['x0'].copy()
    X_nom  = np.tile(x_true, (N + 1, 1)).T
    U_nom  = np.zeros((3, N))

    # Trigger config
    consecutive_threshold = 3
    trigger_counter = 0
    trigger_events = []
    steps_since_last_solve = 0
    crashed = False

    # Logging
    x_hist = np.zeros((6, T_SIM + 1))
    u_hist = np.zeros((3, T_SIM))
    x_ref_hist = np.zeros((6, T_SIM + 1))
    x_hist[:, 0] = x_true
    
    current_wp_idx = 0
    current_x_ref = WAYPOINTS[current_wp_idx]
    x_ref_hist[:, 0] = current_x_ref

    # Initial solve
    X_nom, U_nom, ok = solve_mpc(x_true, X_nom, U_nom, current_x_ref)
    if not ok:
        crashed = True

    for k in range(T_SIM):
        if crashed:
            x_hist[:, k + 1:] = np.nan
            break

        # Check waypoint switching
        dist_to_wp = np.linalg.norm(x_true[0:3] - current_x_ref[0:3])
        if dist_to_wp < 0.2 and current_wp_idx < len(WAYPOINTS) - 1:
            current_wp_idx += 1
            current_x_ref = WAYPOINTS[current_wp_idx]

        x_bar_k = X_nom[:, 0]

        # --- Scenario D: sensor glitch on measured state ---
        measured_x = x_true.copy()
        if cfg['sensor_glitch_step'] is not None and k == cfg['sensor_glitch_step']:
            measured_x[0] -= 0.20   # Fake 20 cm jump away from target for 1 step

        e_k = measured_x - x_bar_k

        # --- Event trigger logic ---
        if controller_type in ['standard', 'full_tube']:
            trigger_counter = consecutive_threshold  # Force trigger every step
        elif use_aux_loop:
            # Formal RPI set membership check
            omega_margin = Omega_H @ e_k - Omega_h
            if np.max(omega_margin) > 0:
                trigger_counter += 1
            else:
                trigger_counter = 0
        else:
            # Heuristic Euclidean trigger (no RPI available)
            pos_error = np.linalg.norm(e_k[0:3])
            if pos_error > 0.15:
                trigger_counter += 1
            else:
                trigger_counter = 0

        if trigger_counter >= consecutive_threshold or steps_since_last_solve >= N - 2:
            trigger_events.append((k, x_true[0:3].copy()))
            X_nom, U_nom, ok = solve_mpc(measured_x, X_nom, U_nom, current_x_ref)
            if not ok and not use_aux_loop:
                crashed = True
            trigger_counter = 0
            steps_since_last_solve = 0
        else:
            steps_since_last_solve += 1

        # --- Control law ---
        u_bar_k = U_nom[:, 0]
        x_bar_k = X_nom[:, 0]
        if use_aux_loop:
            u_true = u_bar_k - K @ (x_true - x_bar_k)
        else:
            u_true = u_bar_k
        u_true = np.clip(u_true, u_lb, u_ub)

        # --- True plant dynamics (with scenario-specific perturbation) ---
        u_applied = u_true * cfg['thrust_scale']    # Scenario B: mass mismatch
        w_wind_val = cfg['w_wind'](k, x_true) if callable(cfg['w_wind']) else cfg['w_wind']
        x_true = A_d @ x_true + B_d @ u_applied + w_wind_val

        # Standard process noise
        d = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1),
                       np.random.uniform(-1.5,1.5)])
        x_true += B_d @ d

        # Log
        x_hist[:, k + 1] = x_true
        u_hist[:, k] = u_true
        x_ref_hist[:, k + 1] = current_x_ref

        # Shift trajectory for next step
        X_nom = np.roll(X_nom, -1, axis=1)
        U_nom = np.roll(U_nom, -1, axis=1)
        if use_aux_loop:
            u_pad = -K @ (X_nom[:, -2] - current_x_ref)
        else:
            u_pad = np.zeros(3)
        U_nom[:, -1] = u_pad
        X_nom[:, -1] = A_d @ X_nom[:, -2] + B_d @ u_pad

    # --- Compute metrics ---
    valid = ~np.isnan(x_hist[0, :])
    valid_hist = x_hist[:, valid]
    valid_ref_hist = x_ref_hist[:, valid]
    if valid_hist.shape[1] > 0:
        pos_errors = np.linalg.norm(
            valid_hist[0:3, :] - valid_ref_hist[0:3, :], axis=0)
        rmse = np.sqrt(np.mean(pos_errors ** 2))
        final_err = np.linalg.norm(valid_hist[0:3, -1] - valid_ref_hist[0:3, -1])

        below = valid_hist < x_lb.reshape(6, 1)
        above = valid_hist > x_ub.reshape(6, 1)
        n_violations = int(np.sum(np.any(below | above, axis=0)))
        excess_above = np.max(valid_hist - x_ub.reshape(6, 1))
        excess_below = np.max(x_lb.reshape(6, 1) - valid_hist)
        max_excess = max(excess_above, excess_below, 0.0)
    else:
        rmse = np.nan; final_err = np.nan; n_violations = 0; max_excess = 0.0

    status = 'Crashed/Infeasible' if crashed else 'Success'

    return {
        'scenario': scenario_name,
        'ctrl_name': ctrl_name,
        'label': cfg['label'],
        'x_hist': x_hist,
        'u_hist': u_hist,
        'triggers': trigger_events,
        'rmse': rmse,
        'final_err': final_err,
        'n_violations': n_violations,
        'max_excess': max_excess,
        'status': status,
    }


if __name__ == '__main__':
    all_results = {}
    for sname in SCENARIOS:
        print(f"Running {SCENARIOS[sname]['label']}...")
        standard = run_simulation(sname, controller_type='standard')
        full_tube = run_simulation(sname, controller_type='full_tube')
        tube = run_simulation(sname, controller_type='et_tube')
        all_results[sname] = (standard, full_tube, tube)

    print("TEST RESULTS")
    print(f"{'Scenario':<28} | {'Controller':<18} | {'Triggers':<9} | "
          f"{'RMSE (m)':<10} | {'Final Err':<10} | {'Bound Viols':<12} | {'Status'}")
    print("-" * 100)
    for sname in SCENARIOS:
        for res in all_results[sname]:
            viol_str = f"{res['n_violations']} steps" if res['n_violations'] > 0 else "NONE ✓"
            print(f"{res['label']:<28} | {res['ctrl_name']:<18} | "
                  f"{len(res['triggers']):<9} | {res['rmse']:<10.4f} | "
                  f"{res['final_err']:<10.4f} | {viol_str:<12} | {res['status']}")
    print("=" * 100)

    # ===== Plotting Configuration =====
    COLORS = {'Standard MPC': '#e74c3c', 'Full Tube MPC': '#8e44ad', 'ET Tube MPC': '#27ae60'}
    STYLES = {'Standard MPC': ':', 'Full Tube MPC': '-.', 'ET Tube MPC': '-'}
    WIDTHS = {'Standard MPC': 1.5, 'Full Tube MPC': 2.0, 'ET Tube MPC': 1.5}
    t_vec = np.arange(T_SIM + 1) * dt
    t_ctrl = np.arange(T_SIM) * dt
    save_dir = base_dir / "plots"
    arena_x, arena_y, arena_z = [x_lb[0], x_ub[0]], [x_lb[1], x_ub[1]], [x_lb[2], x_ub[2]]

    # --- Figure 0: Tube RPI Set Visualization ---
    import pypoman
    from scipy.spatial import ConvexHull
    try:
        Omega_vertices = np.array(pypoman.compute_polytope_vertices(Omega_H, Omega_h))
        pos_vertices = Omega_vertices[:, :3]

        fig0 = plt.figure(figsize=(12, 10))
        
        # 3D Position RPI Set
        ax1 = fig0.add_subplot(221, projection='3d')
        hull_3d = ConvexHull(pos_vertices)
        ax1.scatter(pos_vertices[:, 0], pos_vertices[:, 1], pos_vertices[:, 2], color='#27ae60', s=30, alpha=0.5, label='Vertices')
        for s in hull_3d.simplices:
            s = np.append(s, s[0])
            ax1.plot(pos_vertices[s, 0], pos_vertices[s, 1], pos_vertices[s, 2], color='#27ae60', lw=1.2, alpha=0.7)
        ax1.set_xlabel('Pos Error X (m)')
        ax1.set_ylabel('Pos Error Y (m)')
        ax1.set_zlabel('Pos Error Z (m)')
        ax1.set_title('3D Positional Error Tube (RPI Set)', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.view_init(elev=20, azim=45)

        # Equal aspect ratio in 3D centered on 0, incorporating arena span
        as_x = (x_ub[0] - x_lb[0]) / 2.0
        as_y = (x_ub[1] - x_lb[1]) / 2.0
        as_z = (x_ub[2] - x_lb[2]) / 2.0

        arena_box_edges = [
            ([[-as_x, as_x], [-as_y, -as_y], [-as_z, -as_z]]),
            ([[as_x, as_x], [-as_y, as_y], [-as_z, -as_z]]),
            ([[as_x, -as_x], [as_y, as_y], [-as_z, -as_z]]),
            ([[-as_x, -as_x], [as_y, -as_y], [-as_z, -as_z]]),
            # Top face
            ([[-as_x, as_x], [-as_y, -as_y], [as_z, as_z]]),
            ([[as_x, as_x], [-as_y, as_y], [as_z, as_z]]),
            ([[as_x, -as_x], [as_y, as_y], [as_z, as_z]]),
            ([[-as_x, -as_x], [as_y, as_y], [as_z, as_z]]),
            # Vertical pillars
            ([[-as_x, -as_x], [-as_y, -as_y], [-as_z, as_z]]),
            ([[as_x, as_x], [-as_y, -as_y], [-as_z, as_z]]),
            ([[as_x, as_x], [as_y, as_y], [-as_z, as_z]]),
            ([[-as_x, -as_x], [as_y, as_y], [-as_z, as_z]]),
        ]

        for idx, (ex, ey, ez) in enumerate(arena_box_edges):
            ax1.plot(ex, ey, ez, color='gray', ls='--', lw=1.2, alpha=0.6,
                     label='Arena Span Box' if idx == 0 else "")

        max_range = max(as_x, as_y, as_z)
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        ax1.set_zlim(-max_range, max_range)
        ax1.set_box_aspect((1, 1, 1))
        ax1.legend(loc='upper right', fontsize=8)

        # X-Y Plane Projection
        ax2 = fig0.add_subplot(222)
        hull_xy = ConvexHull(pos_vertices[:, [0, 1]])
        ax2.scatter(pos_vertices[:, 0], pos_vertices[:, 1], color='#27ae60', s=30, alpha=0.5)
        for s in hull_xy.simplices:
            s = np.append(s, s[0])
            ax2.plot(pos_vertices[s, 0], pos_vertices[s, 1], color='#2e7d32', lw=2)
        ax2.set_xlabel('Pos Error X (m)')
        ax2.set_ylabel('Pos Error Y (m)')
        ax2.set_title('X-Y Plane Projection', fontsize=11, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # X-Z Plane Projection
        ax3 = fig0.add_subplot(223)
        hull_xz = ConvexHull(pos_vertices[:, [0, 2]])
        ax3.scatter(pos_vertices[:, 0], pos_vertices[:, 2], color='#27ae60', s=30, alpha=0.5)
        for s in hull_xz.simplices:
            s = np.append(s, s[0])
            ax3.plot(pos_vertices[s, 0], pos_vertices[s, 2], color='#2e7d32', lw=2)
        ax3.set_xlabel('Pos Error X (m)')
        ax3.set_ylabel('Pos Error Z (m)')
        ax3.set_title('X-Z Plane Projection', fontsize=11, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

        # Y-Z Plane Projection
        ax4 = fig0.add_subplot(224)
        hull_yz = ConvexHull(pos_vertices[:, [1, 2]])
        ax4.scatter(pos_vertices[:, 1], pos_vertices[:, 2], color='#27ae60', s=30, alpha=0.5)
        for s in hull_yz.simplices:
            s = np.append(s, s[0])
            ax4.plot(pos_vertices[s, 1], pos_vertices[s, 2], color='#2e7d32', lw=2)
        ax4.set_xlabel('Pos Error Y (m)')
        ax4.set_ylabel('Pos Error Z (m)')
        ax4.set_title('Y-Z Plane Projection', fontsize=11, fontweight='bold')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)

        fig0.suptitle('Robust Positional Invariant (RPI) Tube Set Projection', fontsize=14, fontweight='bold')
        fig0.tight_layout()
        fig0.savefig(save_dir / 'fig0_tube_rpi.png', dpi=200, bbox_inches='tight')
        print("Saved fig0_tube_rpi.png")
    except Exception as e:
        print(f"Could not plot initial tube: {e}")

    # --- Figure 1: 3D Trajectory Comparison (2x2 grid) ---
    fig1 = plt.figure(figsize=(14, 11))
    for idx, sname in enumerate(SCENARIOS):
        standard, full_tube, tube = all_results[sname]
        ax = fig1.add_subplot(2, 2, idx + 1, projection='3d')
        for res in [standard, full_tube, tube]:
            v = ~np.isnan(res['x_hist'][0, :])
            ax.plot(res['x_hist'][0, v], res['x_hist'][1, v], res['x_hist'][2, v],
                    color=COLORS[res['ctrl_name']], ls=STYLES[res['ctrl_name']],
                    lw=WIDTHS[res['ctrl_name']], label=res['ctrl_name'], alpha=0.85)
        # Waypoints
        for i, wp in enumerate(WAYPOINTS):
            ax.scatter(wp[0], wp[1], wp[2], c='purple', s=100, marker='X', zorder=10,
                       label='Waypoints' if i == 0 else None)
        # Start
        ax.scatter(*tube['x_hist'][:3, 0], c='cyan', s=80, marker='o',
                   edgecolors='k', zorder=10, label='Start')
        # Arena wireframe (12 edges of the bounding box)
        for xe, ye, ze in [
            ([arena_x[0], arena_x[1]], [arena_y[0], arena_y[0]], [arena_z[0], arena_z[0]]),
            ([arena_x[1], arena_x[1]], [arena_y[0], arena_y[1]], [arena_z[0], arena_z[0]]),
            ([arena_x[1], arena_x[0]], [arena_y[1], arena_y[1]], [arena_z[0], arena_z[0]]),
            ([arena_x[0], arena_x[0]], [arena_y[1], arena_y[0]], [arena_z[0], arena_z[0]]),
            ([arena_x[0], arena_x[1]], [arena_y[0], arena_y[0]], [arena_z[1], arena_z[1]]),
            ([arena_x[1], arena_x[1]], [arena_y[0], arena_y[1]], [arena_z[1], arena_z[1]]),
            ([arena_x[1], arena_x[0]], [arena_y[1], arena_y[1]], [arena_z[1], arena_z[1]]),
            ([arena_x[0], arena_x[0]], [arena_y[1], arena_y[0]], [arena_z[1], arena_z[1]]),
            ([arena_x[0], arena_x[0]], [arena_y[0], arena_y[0]], [arena_z[0], arena_z[1]]),
            ([arena_x[1], arena_x[1]], [arena_y[0], arena_y[0]], [arena_z[0], arena_z[1]]),
            ([arena_x[1], arena_x[1]], [arena_y[1], arena_y[1]], [arena_z[0], arena_z[1]]),
            ([arena_x[0], arena_x[0]], [arena_y[1], arena_y[1]], [arena_z[0], arena_z[1]]),
        ]:
            ax.plot(xe, ye, ze, color='gray', ls='--', lw=0.8, alpha=0.4)
        ax.set_xlabel('X (m)', fontsize=8); ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_zlabel('Z (m)', fontsize=8)
        ax.set_title(tube['label'], fontweight='bold', fontsize=10)
        ax.set_xlim(arena_x); ax.set_ylim(arena_y); ax.set_zlim(arena_z)
        ax.view_init(elev=22, azim=225)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')
    fig1.suptitle('3D Trajectory Comparison Across Scenarios', fontsize=13, fontweight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(save_dir / 'fig1_3d_trajectory.png', dpi=200, bbox_inches='tight')
    print("Saved fig1_3d_trajectory.png")

    # --- Figure 2: Per-Axis Tracking Error with Tube Bounds (2×2 scenarios, 3 axes each) ---
    # Compute per-axis tube half-widths from Omega (positional RPI bounds)
    # The tight constraint half-widths give the maximum error the tube permits
    hx_tight_ub = hx_tight[:6]   # upper bounds for each state
    hx_tight_lb = -hx_tight[6:]  # lower bounds for each state
    tube_pos_hw = [(x_ub[i] - hx_tight_ub[i] + hx_tight_lb[i] - x_lb[i]) / 2.0
                   for i in range(3)]  # half the tightening margin = tube half-width

    fig2, big_axes = plt.subplots(4, 3, figsize=(14, 14), sharex=True)
    axis_labels = ['X Error (m)', 'Y Error (m)', 'Z Error (m)']
    for row, sname in enumerate(SCENARIOS):
        standard, full_tube, tube = all_results[sname]
        # Build reference history for this scenario
        for col, ax_i in enumerate(range(3)):
            ax = big_axes[row, col]
            for res in [standard, full_tube, tube]:
                v = ~np.isnan(res['x_hist'][0, :])
                ref_h = np.zeros(res['x_hist'].shape[1])
                wp_idx = 0; ref_wp = WAYPOINTS[0][0:3]
                for kk in range(res['x_hist'].shape[1]):
                    xk = res['x_hist'][0:3, kk]
                    if not np.isnan(xk[0]) and wp_idx < len(WAYPOINTS)-1:
                        if np.linalg.norm(xk - ref_wp) < 0.2:
                            wp_idx += 1; ref_wp = WAYPOINTS[wp_idx][0:3]
                    ref_h[kk] = ref_wp[ax_i]
                err_axis = res['x_hist'][ax_i, :] - ref_h
                ax.plot(t_vec[v], err_axis[v], color=COLORS[res['ctrl_name']],
                        ls=STYLES[res['ctrl_name']], lw=WIDTHS[res['ctrl_name']],
                        label=res['ctrl_name'], alpha=0.85)
            # Tube bound shading
            ax.axhspan(-tube_pos_hw[ax_i], tube_pos_hw[ax_i],
                       alpha=0.08, color='#27ae60', label='Tube Bound' if row == 0 and col == 0 else None)
            ax.axhline(-tube_pos_hw[ax_i], color='#27ae60', ls=':', lw=1.0, alpha=0.7)
            ax.axhline( tube_pos_hw[ax_i], color='#27ae60', ls=':', lw=1.0, alpha=0.7)
            ax.axhline(0, color='black', ls='-', lw=0.5, alpha=0.4)
            ax.set_ylabel(axis_labels[col], fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(axis_labels[col], fontweight='bold', fontsize=10)
                if col == 0:
                    ax.legend(fontsize=6, loc='upper right')
            if row == len(SCENARIOS) - 1:
                ax.set_xlabel('Time (s)', fontsize=8)
            if col == 0:
                ax.annotate(SCENARIOS[sname]['label'], xy=(-0.28, 0.5), xycoords='axes fraction',
                            fontsize=8, fontweight='bold', rotation=90, va='center')
    fig2.suptitle('Per-Axis Tracking Error with Tube Bounds (green band = RPI guarantee)',
                  fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig2.savefig(save_dir / 'fig2_axis_error_tube.png', dpi=200, bbox_inches='tight')
    print("Saved fig2_axis_error_tube.png")

    # --- Figure 3: Altitude (Z) Safety — All Scenarios ---
    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for idx, sname in enumerate(SCENARIOS):
        ax = axes3[idx // 2, idx % 2]
        standard, full_tube, tube = all_results[sname]
        for res in [standard, full_tube, tube]:
            v = ~np.isnan(res['x_hist'][0, :])
            ax.plot(t_vec[v], res['x_hist'][2, v],
                    color=COLORS[res['ctrl_name']], ls=STYLES[res['ctrl_name']],
                    lw=WIDTHS[res['ctrl_name']], label=res['ctrl_name'], alpha=0.9)
        # Hard bounds
        ax.axhline(x_lb[2], color='#c0392b', ls='--', lw=1.5, alpha=0.8, label=f'Floor ({x_lb[2]:.1f}m)')
        ax.axhline(x_ub[2], color='#e67e22', ls='--', lw=1.5, alpha=0.8, label=f'Ceiling ({x_ub[2]:.1f}m)')
        # Tightened bounds (what tube MPC actually plans within)
        ax.axhline(hx_tight_lb[2], color='#27ae60', ls=':', lw=1.0, alpha=0.6,
                   label=f'Tight floor ({hx_tight_lb[2]:.2f}m)' if idx == 0 else None)
        ax.axhline(hx_tight_ub[2], color='#27ae60', ls=':', lw=1.0, alpha=0.6,
                   label=f'Tight ceil ({hx_tight_ub[2]:.2f}m)' if idx == 0 else None)
        # Waypoint Z targets
        for i, wp in enumerate(WAYPOINTS):
            ax.axhline(wp[2], color='purple', ls=':', lw=0.9, alpha=0.55,
                       label=f'WP{i+1} Z={wp[2]:.1f}m' if idx == 0 else None)
        ax.set_ylabel('Altitude Z (m)'); ax.set_title(tube['label'], fontweight='bold', fontsize=10)
        ax.set_ylim(x_lb[2] - 0.3, x_ub[2] + 0.3)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right', ncol=2)
    axes3[1, 0].set_xlabel('Time (s)'); axes3[1, 1].set_xlabel('Time (s)')
    fig3.suptitle('Altitude (Z) Safety — Hard Bounds vs Tightened Tube Bounds',
                  fontsize=13, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(save_dir / 'fig3_altitude_safety.png', dpi=200, bbox_inches='tight')
    print("Saved fig3_altitude_safety.png")

    # --- Figure 4: Velocity Smoothness (speed magnitude over time, 2x2) ---
    fig4, axes4 = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for idx, sname in enumerate(SCENARIOS):
        ax = axes4[idx // 2, idx % 2]
        standard, full_tube, tube = all_results[sname]
        for res in [standard, full_tube, tube]:
            v = ~np.isnan(res['x_hist'][0, :])
            speed = np.linalg.norm(res['x_hist'][3:6, v], axis=0)
            ax.plot(t_vec[v], speed, color=COLORS[res['ctrl_name']],
                    ls=STYLES[res['ctrl_name']], lw=WIDTHS[res['ctrl_name']],
                    label=res['ctrl_name'], alpha=0.85)
        # Velocity bound
        v_max = x_ub[3]
        ax.axhline(v_max, color='gray', ls='--', lw=1.2, alpha=0.7, label=f'|v| limit ({v_max:.1f} m/s)')
        ax.set_ylabel('Speed ||v|| (m/s)'); ax.set_title(tube['label'], fontweight='bold', fontsize=10)
        ax.set_ylim(0, v_max + 0.3)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=7)
    axes4[1, 0].set_xlabel('Time (s)'); axes4[1, 1].set_xlabel('Time (s)')
    fig4.suptitle('Velocity Profile — Speed Magnitude Over Time', fontsize=13, fontweight='bold')
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(save_dir / 'fig4_velocity.png', dpi=200, bbox_inches='tight')
    print("Saved fig4_velocity.png")

    # --- Figure 5: Trigger Comparison — All 3 Controllers (4 rows × 3 cols) ---
    ctrl_names_ordered = ['Standard MPC', 'Full Tube MPC', 'ET Tube MPC']
    fig5, axes5 = plt.subplots(len(SCENARIOS), 3, figsize=(15, 9), sharex=True, sharey='row')
    for row, sname in enumerate(SCENARIOS):
        standard, full_tube, tube = all_results[sname]
        results_ordered = [standard, full_tube, tube]
        for col, res in enumerate(results_ordered):
            ax = axes5[row, col]
            n_triggers = len(res['triggers'])
            pct = 100 * n_triggers / T_SIM
            if res['triggers']:
                trig_times = [t[0] * dt for t in res['triggers']]
                ax.eventplot([trig_times], lineoffsets=0.5, linelengths=0.8,
                             colors=COLORS[res['ctrl_name']])
            # Color the background based on solve rate
            bg_alpha = min(0.18, 0.02 + pct / 100 * 0.16)
            ax.set_facecolor(COLORS[res['ctrl_name']] + '22')
            ax.text(0.5, 0.72, f'{n_triggers}/{T_SIM}\n({pct:.0f}%)',
                    transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold',
                    color=COLORS[res['ctrl_name']],
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS[res['ctrl_name']], alpha=0.9))
            ax.set_yticks([])
            ax.grid(True, axis='x', alpha=0.25)
            if row == 0:
                ax.set_title(res['ctrl_name'], fontweight='bold', fontsize=10,
                             color=COLORS[res['ctrl_name']])
            if row == len(SCENARIOS) - 1:
                ax.set_xlabel('Time (s)', fontsize=8)
            if col == 0:
                ax.set_ylabel(SCENARIOS[sname]['label'], fontsize=8, fontweight='bold')
    fig5.suptitle('QP Solve Trigger Timeline — All Controllers × All Scenarios\n'
                  '(Each tick = one MPC solve; % = compute load)',
                  fontsize=12, fontweight='bold')
    fig5.tight_layout(rect=[0, 0, 1, 0.95])
    fig5.savefig(save_dir / 'fig5_trigger_comparison.png', dpi=200, bbox_inches='tight')
    print("Saved fig5_trigger_comparison.png")

    # --- Figure 6: Summary Bar Chart ---
    fig6, axes6 = plt.subplots(1, 3, figsize=(14, 5))
    scenario_labels = [SCENARIOS[s]['label'].split(': ')[1] for s in SCENARIOS]
    ctrl_names = ['Standard MPC', 'Full Tube MPC', 'ET Tube MPC']
    x_pos = np.arange(len(SCENARIOS))
    width = 0.25

    # RMSE
    for ci, cname in enumerate(ctrl_names):
        vals = [all_results[s][ci]['rmse'] for s in SCENARIOS]
        axes6[0].bar(x_pos + ci * width, vals, width, color=COLORS[cname], label=cname)
    axes6[0].set_ylabel('RMSE (m)'); axes6[0].set_title('Tracking RMSE', fontweight='bold')
    axes6[0].set_xticks(x_pos + width); axes6[0].set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=8)
    axes6[0].legend(fontsize=7); axes6[0].grid(True, axis='y', alpha=0.3)

    # Constraint violations
    for ci, cname in enumerate(ctrl_names):
        vals = [all_results[s][ci]['n_violations'] for s in SCENARIOS]
        axes6[1].bar(x_pos + ci * width, vals, width, color=COLORS[cname], label=cname)
    axes6[1].set_ylabel('Violation Steps'); axes6[1].set_title('Constraint Violations', fontweight='bold')
    axes6[1].set_xticks(x_pos + width); axes6[1].set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=8)
    axes6[1].grid(True, axis='y', alpha=0.3)

    # Trigger count
    for ci, cname in enumerate(ctrl_names):
        vals = [len(all_results[s][ci]['triggers']) for s in SCENARIOS]
        axes6[2].bar(x_pos + ci * width, vals, width, color=COLORS[cname], label=cname)
    axes6[2].set_ylabel('QP Solves'); axes6[2].set_title('Optimization Count', fontweight='bold')
    axes6[2].set_xticks(x_pos + width); axes6[2].set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=8)
    axes6[2].grid(True, axis='y', alpha=0.3)

    fig6.suptitle('Performance Summary Across Scenarios', fontsize=13, fontweight='bold')
    fig6.tight_layout(rect=[0, 0, 1, 0.94])
    fig6.savefig(save_dir / 'fig6_summary.png', dpi=200, bbox_inches='tight')
    print("Saved fig6_summary.png")

    print(f"\nAll figures saved to {save_dir}/")
