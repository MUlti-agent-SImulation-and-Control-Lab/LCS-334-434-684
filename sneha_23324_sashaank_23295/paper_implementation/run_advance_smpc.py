# run_advanced_smpc.py
import numpy as np
import time
import matplotlib.pyplot as plt
import config
from env import HighwayEnv
from plot import plot_simulation
from advanced_smpc_controller import AdvancedSMPC

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

def generate_dashboard(log):
    print("Generating Advanced SMPC Performance Dashboard...")
    fig_dash, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig_dash.suptitle("Advanced SMPC: Policy Trees & Risk Allocation", fontsize=16, fontweight='bold')
    
    t = log.t

    # Plot 1: Velocity vs Reference
    axs[0, 0].plot(t, log.v_act, color='blue', lw=2, label='Actual Velocity')
    axs[0, 0].plot(t, log.v_ref, color='black', ls='--', alpha=0.6, label='Target Velocity')
    axs[0, 0].set_title("Ego Velocity Profile")
    axs[0, 0].set_ylabel("Velocity (m/s)")
    axs[0, 0].legend()

    # Plot 2: Lateral Deviation vs Reference
    axs[0, 1].plot(t, log.ey_act, color='green', lw=2, label='Actual ey')
    axs[0, 1].plot(t, log.ey_ref, color='black', ls='--', alpha=0.6, label='Target Lane')
    # Adding typical lane boundaries based on standard config widths
    axs[0, 1].axhline(y=config.LANES[0], color='gray', ls=':', alpha=0.5, label='Lane Centers')
    axs[0, 1].axhline(y=config.LANES[1], color='gray', ls=':', alpha=0.5)
    axs[0, 1].axhline(y=config.LANES[2], color='gray', ls=':', alpha=0.5)
    axs[0, 1].set_title("Lateral Position (ey)")
    axs[0, 1].set_ylabel("Offset from Center (m)")
    axs[0, 1].legend()

    # Plot 3: Commanded Acceleration
    axs[1, 0].step(t, log.a_cmd, where='post', color='red', lw=1.5)
    axs[1, 0].set_title("Commanded Acceleration")
    axs[1, 0].set_ylabel("a (m/s²)")

    # Plot 4: Commanded Steering/Yaw Rate
    axs[1, 1].step(t, log.yaw_dot_cmd, where='post', color='purple', lw=1.5)
    axs[1, 1].set_title("Commanded Steering/Yaw Rate")
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
    plt.savefig('advanced_smpc_dashboard.png')
    print("Dashboard saved successfully as 'advanced_smpc_dashboard.png'!")
    plt.show()

def main():
    # 1. Initialize Environment, Controller, and Logger
    env = HighwayEnv()
    state = env.reset()
    
    # Initialize our Phase 4 Controller (N=20 steps -> 2.0 seconds look-ahead)
    mpc = AdvancedSMPC(horizon=20)
    logger = SimLogger()
    
    # Tracking for visualization
    ego_history = [state["ego"]]
    targets_history = [[t] for t in state["targets"]]
    
    done = False
    step = 0
    max_steps = int(config.SIM_TIME / config.DT)

    print("Running Advanced SMPC with Information Trees & Risk Allocation...")

    # Base goals for the Ego Vehicle
    target_lane = config.LANES[1] # Start by aiming for the center lane
    target_velocity = 20.0

    while step < max_steps and not done:
        t = step * config.DT
        
        # --- HIGH-LEVEL PLANNER ---
        # If TV 1 (in the center lane) is less than 20m ahead, change to Left Lane
        dist_to_tv1 = state["targets"][0][0] - state["ego"][0]
        if 0 < dist_to_tv1 < 20.0:
            # target_lane = config.LANES[2] # Left Lane
            target_lane = config.LANES[0] # right lane 

        # --- FETCH MULTIMODAL PREDICTIONS ---
        # Get the Gaussian Mixture trajectories and probabilities from the TVs
        targets_predictions = []
        for tv in env.targets:
            preds = tv.get_multimodal_predictions(N=mpc.N, dt=config.DT)
            targets_predictions.append(preds)

        # --- ADVANCED SMPC COMPUTES ACTION ---
        # This solves the convex Policy Tree and allocates risk simultaneously!
        t0 = time.time()
        ego_action = mpc.compute_action(
            ego_state=state["ego"],
            targets_predictions=targets_predictions,
            target_lane=target_lane,
            target_vel=target_velocity
        )
        solve_duration = time.time() - t0

        # --- DATA LOGGING ---
        # Extract variables. Assuming state["ego"] = [x, y, v, heading]
        # and ego_action = [accel, steering]
        ey_act = state["ego"][1]
        v_act = state["ego"][2]
        a_cmd = ego_action[0]
        yaw_cmd = ego_action[1]
        
        dists = [np.sqrt((state["ego"][0] - tv[0])**2 + (state["ego"][1] - tv[1])**2) for tv in state["targets"]]
        
        logger.log(t, v_act, target_velocity, ey_act, target_lane, 
                   a_cmd, yaw_cmd, min(dists), solve_duration)

        # --- TRUE ENVIRONMENT EXECUTION ---
        # TV 1 (ahead) keeps its lane. 
        # TV 2 (right lane) actually executes the risky lane change!
        tv1_true_mode = config.MODE_KEEP_LANE
        tv2_true_mode = config.MODE_CHANGE_LEFT if( t > 1.0 and t<4.0 )else config.MODE_KEEP_LANE
        
        tv_true_modes = [tv1_true_mode, tv2_true_mode]

        # Step the actual environment forward
        next_state, done, info = env.step(ego_action, tv_true_modes)
        
        # Record keeping
        ego_history.append(next_state["ego"])
        for i, target_state in enumerate(next_state["targets"]):
            targets_history[i].append(target_state)
            
        state = next_state
        step += 1
        
        if info["collision"]:
            print(f"!!! Collision at t={t:.1f}s !!!")

    print("Advanced SMPC Simulation Complete.")
    
    # Convert to numpy arrays and plot
    ego_history = np.array(ego_history)
    targets_history = [np.array(th) for th in targets_history]
    
    # Plot the top-down trajectory simulation
    plot_simulation(ego_history, targets_history, title="Advanced SMPC: Policy Trees & Risk Allocation")
    
    # Generate and show the 3x2 Performance Dashboard
    generate_dashboard(logger)

if __name__ == "__main__":
    main()