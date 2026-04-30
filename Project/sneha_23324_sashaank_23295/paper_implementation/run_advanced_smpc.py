# run_advanced_smpc.py
import numpy as np
import config
from env import HighwayEnv
from plot import plot_simulation
from advanced_smpc_controller import AdvancedSMPC

def main():
    # 1. Initialize Environment and Controller
    env = HighwayEnv()
    state = env.reset()
    
    # Initialize our Phase 4 Controller (N=20 steps -> 2.0 seconds look-ahead)
    mpc = AdvancedSMPC(horizon=20)
    
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
        ego_action = mpc.compute_action(
            ego_state=state["ego"],
            targets_predictions=targets_predictions,
            target_lane=target_lane,
            target_vel=target_velocity
        )

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
    
    plot_simulation(ego_history, targets_history, title="Advanced SMPC: Policy Trees & Risk Allocation")

if __name__ == "__main__":
    main()