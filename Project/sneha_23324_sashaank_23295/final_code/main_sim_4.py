import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# --- Your Custom Modules ---
import config
import track
from dynamics import FrenetVehicle
from smpc_controller import SMPCController

# ==========================================
# 1. DATA LOGGING SYSTEM (WITH REFERENCES)
# ==========================================
class SimLogger:
    def __init__(self):
        self.t = []
        self.v_act = []
        self.v_ref = []
        self.ey_act = []
        self.ey_ref = []
        self.a_cmd = []
        self.yaw_dot_cmd = []
        self.min_dist = []
        self.solve_times = []

    def log(self, t, v, v_ref, ey, ey_ref, a, yaw_dot, dist, s_time):
        self.t.append(t)
        self.v_act.append(v)
        self.v_ref.append(v_ref)
        self.ey_act.append(ey)
        self.ey_ref.append(ey_ref)
        self.a_cmd.append(a)
        self.yaw_dot_cmd.append(yaw_dot)
        self.min_dist.append(dist)
        self.solve_times.append(s_time)

logger = SimLogger()

# ==========================================
# 2. SCENARIO INITIALIZATION (ROADBLOCK)
# ==========================================
ego = FrenetVehicle('Ego', s_init=10.0, ey_init=0.0, v_init=15.0, color='blue')

# Existing dynamic traffic
tv1 = FrenetVehicle('TV1', s_init=40.0, ey_init=0.0, v_init=8.0, color='red')
tv2 = FrenetVehicle('TV2', s_init=30.0, ey_init=config.LANE_WIDTH, v_init=14.0, color='orange')
tv3 = FrenetVehicle('TV3', s_init=60.0, ey_init=-config.LANE_WIDTH, v_init=12.0, color='green')

# THE ROADBLOCK (All 3 lanes blocked further down the track)
tv4 = FrenetVehicle('TV4', s_init=100.0, ey_init=config.LANE_WIDTH, v_init=10.0, color='purple')
tv5 = FrenetVehicle('TV5', s_init=100.0, ey_init=0.0, v_init=10.0, color='magenta')
tv6 = FrenetVehicle('TV6', s_init=100.0, ey_init=-config.LANE_WIDTH, v_init=10.0, color='brown')

tvs = [tv1, tv2, tv3, tv4, tv5, tv6]
all_vehicles = [ego] + tvs

# Initialize the SMPC Controller
mpc = SMPCController()

# Set Reference Targets for the Ego Vehicle
TARGET_VELOCITY = 15.0
TARGET_LANE = 0.0

# ==========================================
# 3. ANIMATION PLOT SETUP
# ==========================================
fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
ax_anim.set_aspect('equal')
ax_anim.set_title("Live Vehicle Simulation (Heavy Traffic / Roadblock)")
ax_anim.set_xlabel("X (meters)")
ax_anim.set_ylabel("Y (meters)")

# Draw the Track
s_vals = [i * (config.TRACK_LENGTH / 500) for i in range(500)]
for offset in [-1.5 * config.LANE_WIDTH, -0.5 * config.LANE_WIDTH, 0.5 * config.LANE_WIDTH, 1.5 * config.LANE_WIDTH]:
    line_style = 'k-' if abs(offset) == 1.5 * config.LANE_WIDTH else 'k--'
    x_vals, y_vals = zip(*[track.get_cartesian(s, offset)[:2] for s in s_vals])
    ax_anim.plot(x_vals, y_vals, line_style, alpha=0.5)

scatters, texts = [], []
pred_lines = []
# Increased pool to 20 to handle up to 6 TVs with multiple modes
for _ in range(20):
    line, = ax_anim.plot([], [], '--', linewidth=2, alpha=0.5, zorder=4)
    pred_lines.append(line)

for veh in all_vehicles:
    sc, = ax_anim.plot([], [], 'o', color=veh.color, markersize=12, zorder=5)
    txt = ax_anim.text(0, 0, veh.id, fontsize=9, fontweight='bold', ha='left')
    scatters.append(sc)
    texts.append(txt)

# ==========================================
# 4. ANIMATION UPDATE LOOP
# ==========================================
def init():
    for sc, txt in zip(scatters, texts):
        sc.set_data([], [])
        txt.set_position((0, 0))
    for line in pred_lines:
        line.set_data([], [])
    return scatters + texts + pred_lines

def update(frame):
    current_time = frame * config.DT
    
    for i, veh in enumerate(all_vehicles):
        
        # --- SMPC FOR EGO VEHICLE ---
        if veh.id == 'Ego':
            current_state = np.array([veh.s, veh.ey, veh.epsi, veh.v])
            tv_predictions = []
            line_idx = 0 
            
            for other_veh in all_vehicles:
                if other_veh.id != 'Ego':
                    gmm_modes = other_veh.get_gmm_prediction(mpc.N, config.DT)
                    tv_predictions.append(gmm_modes)
                    
                    # Draw Predictions safely up to available line pool
                    for mode in gmm_modes:
                        if line_idx < len(pred_lines):
                            traj = mode['trajectory']
                            pred_x, pred_y = [], []
                            for k in range(mpc.N):
                                px, py, _ = track.get_cartesian(traj[k, 0], traj[k, 1])
                                pred_x.append(px)
                                pred_y.append(py)
                            
                            alpha_val = 0.8 if mode['weight'] > 0.5 else 0.3
                            pred_lines[line_idx].set_data(pred_x, pred_y)
                            pred_lines[line_idx].set_color(other_veh.color)
                            pred_lines[line_idx].set_alpha(alpha_val)
                            line_idx += 1
                        
            # Hide unused prediction lines
            for j in range(line_idx, len(pred_lines)):
                pred_lines[j].set_data([], [])
            
            # --- SOLVE MPC & TRACK TIME ---
            t0 = time.time()
            a_cmd, epsi_dot_cmd = mpc.solve(current_state, tv_predictions)
            solve_duration = time.time() - t0
            
            # Feedforward Curvature
            kappa = veh.get_curvature(veh.s)
            denominator = max(0.1, 1.0 - veh.ey * kappa)
            base_yaw_rate = (veh.v / denominator) * kappa
            yaw_rate_cmd = epsi_dot_cmd + base_yaw_rate
            
            # --- LOG EGO DATA WITH REFERENCES ---
            dists = [np.sqrt((veh.s - tv.s)**2 + (veh.ey - tv.ey)**2) for tv in tvs]
            logger.log(current_time, veh.v, TARGET_VELOCITY, veh.ey, TARGET_LANE, 
                       a_cmd, yaw_rate_cmd, min(dists), solve_duration)
            
        # --- DUMMY CONTROLLER FOR TVs ---
        else:
            a_cmd = 0.0 
            kappa = veh.get_curvature(veh.s)
            denominator = max(0.1, 1.0 - veh.ey * kappa)
            yaw_rate_cmd = (veh.v / denominator) * kappa 
        
        # Update physics
        veh.update(a_cmd, yaw_rate_cmd, config.DT)
        
        # Update visuals
        x, y, _ = track.get_cartesian(veh.s, veh.ey)
        scatters[i].set_data([x], [y])
        texts[i].set_position((x + 2, y + 2))

    return scatters + texts + pred_lines

# ==========================================
# 5. RUN & SAVE ANIMATION
# ==========================================
print("Rendering Heavy Traffic simulation. This will populate the logger...")
ani = animation.FuncAnimation(fig_anim, update, frames=config.SIM_FRAMES, 
                              init_func=init, blit=True, 
                              interval=config.DT*1000, repeat=False)

ani.save('heavy_traffic_smpc.gif', writer='pillow', fps=int(1/config.DT))
print("Animation saved successfully as 'heavy_traffic_smpc.gif'!")
plt.close(fig_anim)

# ==========================================
# 6. GENERATE PERFORMANCE DASHBOARD
# ==========================================
def generate_dashboard(log):
    print("Generating Performance Dashboard...")
    fig_dash, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig_dash.suptitle("SMPC Simulation Analysis: Heavy Traffic Roadblock", fontsize=16, fontweight='bold')
    
    t = log.t

    # Plot 1: Velocity vs Reference
    axs[0, 0].plot(t, log.v_act, color='blue', lw=2, label='Actual Velocity')
    axs[0, 0].plot(t, log.v_ref, color='black', ls='--', alpha=0.6, label='Target Velocity')
    axs[0, 0].set_title("Ego Velocity Profile")
    axs[0, 0].set_ylabel("Velocity (m/s)")
    axs[0, 0].legend()

    # Plot 2: Lateral Deviation vs Reference
    axs[0, 1].plot(t, log.ey_act, color='green', lw=2, label='Actual ey')
    axs[0, 1].plot(t, log.ey_ref, color='black', ls='--', alpha=0.6, label='Centerline')
    axs[0, 1].axhline(y=config.LANE_WIDTH, color='gray', ls=':', alpha=0.5, label='Lane Bounds')
    axs[0, 1].axhline(y=-config.LANE_WIDTH, color='gray', ls=':', alpha=0.5)
    axs[0, 1].set_title("Lateral Position (ey)")
    axs[0, 1].set_ylabel("Offset from Center (m)")
    axs[0, 1].legend()

    # Plot 3: Commanded Acceleration
    axs[1, 0].step(t, log.a_cmd, where='post', color='red', lw=1.5)
    axs[1, 0].set_title("Commanded Acceleration (MPC Output)")
    axs[1, 0].set_ylabel("a (m/s²)")

    # Plot 4: Commanded Yaw Rate
    axs[1, 1].step(t, log.yaw_dot_cmd, where='post', color='purple', lw=1.5)
    axs[1, 1].set_title("Commanded Yaw Rate (Total)")
    axs[1, 1].set_ylabel("rad/s")

    # Plot 5: Safety Distance
    axs[2, 0].plot(t, log.min_dist, color='orange', lw=2)
    axs[2, 0].axhline(y=3.0, color='red', ls=':', label='Safety Threshold')
    axs[2, 0].set_title("Proximity to Traffic")
    axs[2, 0].set_ylabel("Distance (m)")
    axs[2, 0].legend()

    # Plot 6: Solver Time
    axs[2, 1].bar(t, log.solve_times, width=config.DT*0.8, color='gray', alpha=0.7)
    axs[2, 1].axhline(y=config.DT, color='red', ls='--', label='Real-time Limit (DT)')
    axs[2, 1].set_title("Solver Computation Time")
    axs[2, 1].set_ylabel("Seconds")
    axs[2, 1].legend()

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('heavy_traffic_dashboard.png')
    print("Dashboard saved successfully as 'heavy_traffic_dashboard.png'!")
    plt.show()

# Run the dashboard generation
generate_dashboard(logger)