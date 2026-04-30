# run.py
import numpy as np
import cartersian_coordinates.config as config
from cartersian_coordinates.env import HighwayEnv
from cartersian_coordinates.plot import plot_simulation

def run_test_simulation():
    # Initialize Environment
    env = HighwayEnv()
    state = env.reset()
    
    # Data storage for visualization
    # state["ego"] is [x, y, v]
    ego_history = [state["ego"]]
    
    # state["targets"] is list of [x, y, v]
    targets_history = [[t] for t in state["targets"]]
    
    total_reward = 0
    done = False
    step = 0
    max_steps = int(config.SIM_TIME / config.DT)

    print("Starting Dummy Simulation...")
    print(f"Ego initial y: {state['ego'][1]} (Center Lane)")

    while step < max_steps and not done:
        t_current = step * config.DT
        
        # --- DEFINE DUMMY ACTIONS (OPEN LOOP) ---
        
        # 1. Ego Action: [accel, dy_dt]
        # Drive straight for 2 seconds, then shift to Left Lane
        if t_current < 2.0:
            ego_accel = 0.0 # Maintain speed
            ego_dy_dt = 0.0 # Stay in lane
        elif 2.0 <= t_current < 6.0:
            ego_accel = 0.5 # Accelerate slightly
            # Compute dy/dt needed to reach left lane (y=3.5) over 4 seconds
            # dy = y_target - y_curr = 3.5 - 0.0 = 3.5
            # dt = 4.0
            # dy_dt = 3.5 / 4.0 = 0.875
            ego_dy_dt = 0.875 
        else:
            ego_accel = 0.0
            ego_dy_dt = 0.0 # Hold left lane
            
        ego_action = np.array([ego_accel, ego_dy_dt])

        # 2. Target Actions: list of [accel, dy_dt]
        # Target 1 (in center lane): Constant Velocity
        t1_action = np.array([0.0, 0.0])
        
        # Target 2 (in left lane): Braking hard, staying in lane
        # Ego will eventually overtake it after lane change
        t2_action = np.array([-1.5, 0.0]) 
        
        target_actions = [t1_action, t2_action]

        # --- STEP THE ENVIRONMENT ---
        next_state, reward, done, info = env.step(ego_action, target_actions)
        
        # --- SAVE HISTORY ---
        ego_history.append(next_state["ego"])
        for i, t in enumerate(next_state["targets"]):
            targets_history[i].append(t)
            
        total_reward += reward
        step += 1
        
        if info["collision"]:
            print(f"!!! COLLISION DETECTED at time {t_current:.2f} s !!!")

    print(f"Simulation ended. Total Steps: {step}, Total Reward: {total_reward:.2f}")

    # --- CONVERT TO NUMPY ARRAYS FOR PLOTTING ---
    ego_history = np.array(ego_history)
    targets_history = [np.array(th) for th in targets_history]

    # --- VISUALIZE ---
    plot_simulation(ego_history, targets_history, title="Scenario Check: Ego Lane Change")

if __name__ == "__main__":
    run_test_simulation()