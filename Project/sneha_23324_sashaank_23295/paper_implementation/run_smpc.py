# run_smpc.py
import numpy as np
import config
from env import HighwayEnv
from plot import plot_simulation
from smpc_controller import StochasticMPC

def main():
    # 1. Initialize Environment and Controller
    env = HighwayEnv()
    state = env.reset()
    
    mpc = StochasticMPC(horizon=20) # 2.0 second look-ahead at DT=0.1
    
    # 2. Tracking for visualization
    ego_history = [state["ego"]]
    targets_history = [[t] for t in state["targets"]]
    
    done = False
    step = 0
    max_steps = int(config.SIM_TIME / config.DT)

    print("Running Stochastic MPC (SMPC) Environment...")

    # Base goals for the Ego Vehicle
    target_lane = config.LANES[1] # Aim for the center lane
    target_velocity = 20.0

    while step < max_steps and not done:
        t = step * config.DT
        
        # --- 3. FETCH MULTIMODAL PREDICTIONS ---
        # Instead of just taking the current state, we ask the TV objects 
        # to generate their probabilistic futures over the MPC horizon (N).
        targets_predictions = []
        for tv in env.targets:
            preds = tv.get_multimodal_predictions(N=mpc.N, dt=config.DT)
            targets_predictions.append(preds)

        # --- 4. SMPC COMPUTES ACTION ---
        ego_action = mpc.compute_action(
            ego_state=state["ego"],
            targets_predictions=targets_predictions, # Passing predictions, not states!
            target_lane=target_lane,
            target_vel=target_velocity
        )

        # --- 5. TRUE ENVIRONMENT UPDATE ---
        # The TVs still need to execute an *actual* mode in the real world.
        # TV 1 (ahead) keeps its lane. 
        # TV 2 (right lane) actually executes the lane change it predicted at 20% probability!
        tv1_true_mode = config.MODE_KEEP_LANE
        tv2_true_mode = config.MODE_CHANGE_LEFT if t > 1.0 else config.MODE_KEEP_LANE
        
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

    print("SMPC Simulation Complete.")
    
    # Convert to numpy arrays and plot
    ego_history = np.array(ego_history)
    targets_history = [np.array(th) for th in targets_history]
    
    plot_simulation(ego_history, targets_history, title="Stochastic MPC with Multimodal Predictions")

if __name__ == "__main__":
    main()