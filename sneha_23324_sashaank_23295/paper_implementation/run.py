# run.py
import numpy as np
import config
from env import HighwayEnv
from plot import plot_simulation
from mpc_controller import DeterministicMPC

def main():
    env = HighwayEnv()
    state = env.reset()
    
    mpc = DeterministicMPC(horizon=20)
    
    ego_history = [state["ego"]]
    targets_history = [[t] for t in state["targets"]]
    
    done = False
    step = 0
    max_steps = int(config.SIM_TIME / config.DT)

    print("Running MPC-Controlled Frenet Environment...")

    # High-level goals
    target_lane = config.LANES[1] # Start aiming for center lane
    target_velocity = 20.0

    while step < max_steps and not done:
        t = step * config.DT
        
        # --- High Level Planner ---
        # If TV_1 (in center lane) is less than 20m ahead, change to Left Lane
        dist_to_tv1 = state["targets"][0][0] - state["ego"][0]
        if 0 < dist_to_tv1 < 20.0:
            target_lane = config.LANES[2] # Left Lane

        # --- MPC computes action ---
        ego_action = mpc.compute_action(
            ego_state=state["ego"],
            target_states=state["targets"],
            target_lane=target_lane,
            target_vel=target_velocity
        )

        # --- TV True Modes ---
        # Both TVs just keep their lane and speed for the deterministic test
        tv_modes = [config.MODE_KEEP_LANE, config.MODE_KEEP_LANE]

        # Step Environment
        next_state, done, info = env.step(ego_action, tv_modes)
        
        ego_history.append(next_state["ego"])
        for i, target_state in enumerate(next_state["targets"]):
            targets_history[i].append(target_state)
            
        state = next_state
        step += 1
        
        if info["collision"]:
            print(f"Collision at t={t:.1f}s!")

    print("Simulation Complete.")
    
    ego_history = np.array(ego_history)
    targets_history = [np.array(th) for th in targets_history]
    plot_simulation(ego_history, targets_history, title="Deterministic MPC Autonomous Lane Change")

if __name__ == "__main__":
    main()