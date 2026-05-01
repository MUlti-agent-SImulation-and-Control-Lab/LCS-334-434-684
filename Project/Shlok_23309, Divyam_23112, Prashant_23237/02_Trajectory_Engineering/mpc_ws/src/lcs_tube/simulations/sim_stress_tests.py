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
    WIDTHS = {'Standard MPC': 1.5, 'Full Tube MPC': 1.5, 'ET Tube MPC': 2.0}
    t_vec = np.arange(T_SIM + 1) * dt
    t_ctrl = np.arange(T_SIM) * dt
    save_dir = base_dir
    arena_x, arena_y, arena_z = [x_lb[0], x_ub[0]], [x_lb[1], x_ub[1]], [x_lb[2], x_ub[2]]

    # --- Figure 1: XY Trajectory (2x2 grid) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    for idx, sname in enumerate(SCENARIOS):
        ax = axes1[idx // 2, idx % 2]
        standard, full_tube, tube = all_results[sname]
        # Arena boundary
        rect = plt.Rectangle((arena_x[0], arena_y[0]), arena_x[1]-arena_x[0],
                              arena_y[1]-arena_y[0], fill=False, ec='gray',
                              ls='--', lw=1.5, label='Arena')
        ax.add_patch(rect)
        for res in [standard, full_tube, tube]:
            v = ~np.isnan(res['x_hist'][0, :])
            ax.plot(res['x_hist'][0, v], res['x_hist'][1, v],
                    color=COLORS[res['ctrl_name']], ls=STYLES[res['ctrl_name']],
                    lw=WIDTHS[res['ctrl_name']], label=res['ctrl_name'])
        for i, wp in enumerate(WAYPOINTS):
            ax.scatter(wp[0], wp[1], c='purple', s=120, marker='X', zorder=10,
                       label='Waypoints' if i == 0 else None)
        ax.scatter(tube['x_hist'][0, 0], tube['x_hist'][1, 0], c='cyan', s=80,
                   marker='o', edgecolors='k', zorder=10, label='Start')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(tube['label'], fontweight='bold', fontsize=10)
        ax.set_xlim(arena_x[0]-0.3, arena_x[1]+0.3)
        ax.set_ylim(arena_y[0]-0.3, arena_y[1]+0.3)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=7, loc='upper right')
    fig1.suptitle('XY Trajectory Comparison', fontsize=13, fontweight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(save_dir / 'fig1_xy_trajectory.png', dpi=200, bbox_inches='tight')
    print("Saved fig1_xy_trajectory.png")

    # --- Figure 2: Position Error Over Time (2x2 grid) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for idx, sname in enumerate(SCENARIOS):
        ax = axes2[idx // 2, idx % 2]
        standard, full_tube, tube = all_results[sname]
        for res in [standard, full_tube, tube]:
            v = ~np.isnan(res['x_hist'][0, :])
            pos_err = np.linalg.norm(res['x_hist'][0:3, v] -
                                     res['x_hist'][0:3, 0:1], axis=0)
            # Compute error vs reference
            ref_h = np.zeros((3, res['x_hist'].shape[1]))
            wp_idx = 0; ref_wp = WAYPOINTS[0][0:3]
            for kk in range(res['x_hist'].shape[1]):
                xk = res['x_hist'][0:3, kk]
                if not np.isnan(xk[0]):
                    if wp_idx < len(WAYPOINTS)-1 and np.linalg.norm(xk - ref_wp) < 0.2:
                        wp_idx += 1; ref_wp = WAYPOINTS[wp_idx][0:3]
                ref_h[:, kk] = ref_wp
            err = np.linalg.norm(res['x_hist'][0:3, v] - ref_h[:, v], axis=0)
            ax.plot(t_vec[:len(err)], err, color=COLORS[res['ctrl_name']],
                    ls=STYLES[res['ctrl_name']], lw=WIDTHS[res['ctrl_name']],
                    label=res['ctrl_name'])
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Position Error (m)')
        ax.set_title(tube['label'], fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=7)
    fig2.suptitle('Position Tracking Error', fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(save_dir / 'fig2_position_error.png', dpi=200, bbox_inches='tight')
    print("Saved fig2_position_error.png")

    # --- Figure 3: Per-Axis Position Over Time (one scenario: A_wind) ---
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axis_labels = ['X (m)', 'Y (m)', 'Z (m)']
    sname_demo = 'A_wind'
    standard, full_tube, tube = all_results[sname_demo]
    for ax_i, ax in enumerate(axes3):
        for res in [standard, full_tube, tube]:
            v = ~np.isnan(res['x_hist'][0, :])
            ax.plot(t_vec[v], res['x_hist'][ax_i, v], color=COLORS[res['ctrl_name']],
                    ls=STYLES[res['ctrl_name']], lw=WIDTHS[res['ctrl_name']],
                    label=res['ctrl_name'])
        ax.axhline(x_lb[ax_i], color='gray', ls='--', lw=1, alpha=0.7, label='Bounds' if ax_i==0 else None)
        ax.axhline(x_ub[ax_i], color='gray', ls='--', lw=1, alpha=0.7)
        # Mark waypoints
        for wp in WAYPOINTS:
            ax.axhline(wp[ax_i], color='purple', ls=':', lw=0.8, alpha=0.5)
        ax.set_ylabel(axis_labels[ax_i]); ax.grid(True, alpha=0.3)
        if ax_i == 0: ax.legend(fontsize=7, ncol=4, loc='upper right')
    axes3[-1].set_xlabel('Time (s)')
    fig3.suptitle(f'Per-Axis Position — {SCENARIOS[sname_demo]["label"]}',
                  fontsize=13, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig(save_dir / 'fig3_per_axis.png', dpi=200, bbox_inches='tight')
    print("Saved fig3_per_axis.png")

    # --- Figure 4: Control Effort (2x2 grid, norm of u) ---
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 8))
    for idx, sname in enumerate(SCENARIOS):
        ax = axes4[idx // 2, idx % 2]
        standard, full_tube, tube = all_results[sname]
        for res in [standard, full_tube, tube]:
            v = ~np.isnan(res['u_hist'][0, :])
            u_norm = np.linalg.norm(res['u_hist'][:, v], axis=0)
            ax.plot(t_ctrl[:len(u_norm)], u_norm, color=COLORS[res['ctrl_name']],
                    ls=STYLES[res['ctrl_name']], lw=WIDTHS[res['ctrl_name']],
                    label=res['ctrl_name'], alpha=0.8)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('||u|| (m/s²)')
        ax.set_title(tube['label'], fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=7)
    fig4.suptitle('Control Effort Magnitude', fontsize=13, fontweight='bold')
    fig4.tight_layout(rect=[0, 0, 1, 0.96])
    fig4.savefig(save_dir / 'fig4_control_effort.png', dpi=200, bbox_inches='tight')
    print("Saved fig4_control_effort.png")

    # --- Figure 5: Event Trigger Timeline ---
    fig5, axes5 = plt.subplots(len(SCENARIOS), 1, figsize=(12, 8), sharex=True)
    for idx, sname in enumerate(SCENARIOS):
        ax = axes5[idx]
        standard, full_tube, tube = all_results[sname]
        # ET Tube triggers
        if tube['triggers']:
            trig_times = [t[0] * dt for t in tube['triggers']]
            ax.eventplot([trig_times], lineoffsets=0.5, linelengths=0.8,
                         colors=COLORS['ET Tube MPC'], label='ET Tube MPC')
        n_et = len(tube['triggers'])
        n_total = T_SIM
        ax.text(0.98, 0.7, f'ET: {n_et}/{n_total} solves ({100*n_et/n_total:.0f}%)',
                transform=ax.transAxes, ha='right', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
        ax.set_ylabel(SCENARIOS[sname]['label'].split(':')[0], fontweight='bold', fontsize=9)
        ax.set_yticks([]); ax.grid(True, axis='x', alpha=0.3)
    axes5[-1].set_xlabel('Time (s)')
    fig5.suptitle('Event Trigger Timeline (ET Tube MPC)', fontsize=13, fontweight='bold')
    fig5.tight_layout(rect=[0, 0, 1, 0.96])
    fig5.savefig(save_dir / 'fig5_triggers.png', dpi=200, bbox_inches='tight')
    print("Saved fig5_triggers.png")

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
