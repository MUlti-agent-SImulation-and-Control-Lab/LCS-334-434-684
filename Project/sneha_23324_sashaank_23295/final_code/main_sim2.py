import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import config
import track
from dynamics import FrenetVehicle
from smpc_controller import SMPCController # <-- Import the Brain

# --- 1. Initialize Scenario ---
# 1. Initialization
ego = FrenetVehicle('Ego', s_init=10.0, ey_init=0.0, v_init=15.0, color='blue')

# Existing dynamic traffic
tv1 = FrenetVehicle('TV1', s_init=40.0, ey_init=0.0, v_init=8.0, color='red')
tv2 = FrenetVehicle('TV2', s_init=30.0, ey_init=config.LANE_WIDTH, v_init=14.0, color='orange')
tv3 = FrenetVehicle('TV3', s_init=60.0, ey_init=-config.LANE_WIDTH, v_init=12.0, color='green')

# THE ROADBLOCK (All 3 lanes blocked further down the track)
tv4 = FrenetVehicle('TV4', s_init=100.0, ey_init=config.LANE_WIDTH, v_init=10.0, color='purple') # Left
tv5 = FrenetVehicle('TV5', s_init=100.0, ey_init=0.0, v_init=10.0, color='magenta')             # Center
tv6 = FrenetVehicle('TV6', s_init=100.0, ey_init=-config.LANE_WIDTH, v_init=10.0, color='brown')  # Right

tvs = [tv1, tv2, tv3, tv4, tv5, tv6]
all_vehicles = [ego] + tvs

# Initialize the SMPC Controller
mpc = SMPCController()

# --- 2. Plot Setup ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
ax.set_title("Live Vehicle Simulation (Nominal SMPC)")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")

s_vals = [i * (config.TRACK_LENGTH / 500) for i in range(500)]
for offset in [-1.5 * config.LANE_WIDTH, -0.5 * config.LANE_WIDTH, 0.5 * config.LANE_WIDTH, 1.5 * config.LANE_WIDTH]:
    line_style = 'k-' if abs(offset) == 1.5 * config.LANE_WIDTH else 'k--'
    x_vals, y_vals = zip(*[track.get_cartesian(s, offset)[:2] for s in s_vals])
    ax.plot(x_vals, y_vals, line_style, alpha=0.5)

scatters, texts = [], []
# Create a pool of line objects for drawing GMM predictions
# Max 4 vehicles * Max 2 modes = 8 lines max. We'll pre-allocate 10 just in case.
pred_lines = []
for _ in range(10):
    line, = ax.plot([], [], '--', linewidth=2, alpha=0.5, zorder=4)
    pred_lines.append(line)

for veh in all_vehicles:
    sc, = ax.plot([], [], 'o', color=veh.color, markersize=12, zorder=5)
    txt = ax.text(0, 0, veh.id, fontsize=9, fontweight='bold', ha='left')
    scatters.append(sc)
    texts.append(txt)

# --- 3. Animation Update Loop ---
def init():
    for sc, txt in zip(scatters, texts):
        sc.set_data([], [])
        txt.set_position((0, 0))
    for line in pred_lines:
        line.set_data([], [])
    return scatters + texts + pred_lines

def update(frame):
    for i, veh in enumerate(all_vehicles):
        
       # --- SMPC FOR EGO VEHICLE ---
        if veh.id == 'Ego':
            current_state = np.array([veh.s, veh.ey, veh.epsi, veh.v])
            
            # Gather GMM predictions from all Target Vehicles
            tv_predictions = []
            line_idx = 0 # To track which prediction line we are drawing
            
            for other_veh in all_vehicles:
                if other_veh.id != 'Ego':
                    gmm_modes = other_veh.get_gmm_prediction(mpc.N, config.DT)
                    
                    # Pass the ENTIRE list of modes to the controller
                    tv_predictions.append(gmm_modes)
                    
                    # --- DRAW THE FUTURE PREDICTIONS ---
                    for mode in gmm_modes:
                        traj = mode['trajectory']
                        # Convert frenet trajectory to cartesian for plotting
                        pred_x, pred_y = [], []
                        for k in range(mpc.N):
                            px, py, _ = track.get_cartesian(traj[k, 0], traj[k, 1])
                            pred_x.append(px)
                            pred_y.append(py)
                        
                        # Make lower probability modes slightly more transparent
                        alpha_val = 0.8 if mode['weight'] > 0.5 else 0.3
                        pred_lines[line_idx].set_data(pred_x, pred_y)
                        pred_lines[line_idx].set_color(other_veh.color)
                        pred_lines[line_idx].set_alpha(alpha_val)
                        line_idx += 1
                        
            # Hide any unused prediction lines
            for j in range(line_idx, len(pred_lines)):
                pred_lines[j].set_data([], [])
            
            # The MPC outputs a CORRECTIONAL acceleration and yaw rate
            a_cmd, epsi_dot_cmd = mpc.solve(current_state, tv_predictions)
            
           
            # --- FEEDFORWARD CURVATURE ---
            # Calculate the base steering needed just to follow the curve
            kappa = veh.get_curvature(veh.s)
            denominator = 1.0 - veh.ey * kappa
            if denominator < 0.1: denominator = 0.1
            base_yaw_rate = (veh.v / denominator) * kappa
            
            # Final command = Correction (MPC) + Base Curve (Feedforward)
            yaw_rate_cmd = epsi_dot_cmd + base_yaw_rate
            
        # --- DUMMY CONTROLLER FOR TVs ---
        else:
            a_cmd = 0.0 
            kappa = veh.get_curvature(veh.s)
            denominator = 1.0 - veh.ey * kappa
            if denominator < 0.1: denominator = 0.1
            yaw_rate_cmd = (veh.v / denominator) * kappa 
        
        # Update vehicle physics
        veh.update(a_cmd, yaw_rate_cmd, config.DT)
        
        # Visuals
        x, y, _ = track.get_cartesian(veh.s, veh.ey)
        scatters[i].set_data([x], [y])
        texts[i].set_position((x + 2, y + 2))

    return scatters + texts + pred_lines

# Run Animation
ani = animation.FuncAnimation(fig, update, frames=config.SIM_FRAMES, 
                              init_func=init, blit=True, 
                              interval=config.DT*1000, repeat=False)

print("Rendering Nominal MPC simulation...")
ani.save('nominal_smpc_simulation.gif', writer='pillow', fps=int(1/config.DT))
print("Saved successfully!")
plt.show()