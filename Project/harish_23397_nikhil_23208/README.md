# Self-Balancing Robot Project (LCS-334/434/684)

This project simulates a self-balancing two-wheeled robot using MATLAB, Simulink, and CoppeliaSim. The controller stabilizes the robot's tilt angle while allowing manual movement.

## Repository Structure
- `src/`: 
    - `coppeliasimfile.ttt`: The main CoppeliaSim scene file.
    - `matlab/`: 
        - `main.slx`: Simulink model containing the control block diagram and scopes.
        - `main2.m`: MATLAB script to visualize PID output plots.
        - `self_balancing_bot_matlab.m`: Script for autonomous forward and backward movement.
- `results/`: Contains existing plots and simulation recordings for verification.
- `report/`: Contains the final project report (PDF).

## Prerequisites
- MATLAB & Simulink (with Control System Toolbox).
- CoppeliaSim (V4.6 or later).

## Step-by-Step Instructions to Run

### 1. Manual Control in CoppeliaSim
1. Open **CoppeliaSim**.
2. Go to `File > Open Scene...` and select `src/coppeliasimfile.ttt`.
3. Press the **Play** button in CoppeliaSim.
4. **To move the robot:** 
   - Click on the **Robot body** in the scene to select it.
   - Use the **Keyboard Arrow Keys** (Up, Down, Left, Right) to control the movement.

### 2. Running the Simulink Model (Plots & Matrices)
1. Open **MATLAB** and navigate to the `src/matlab/` directory.
2. Open `main.slx`.
3. Press the **Run** button in Simulink.
4. **To view results:**
   - **Position Plot:** Double-click the **Position Scope** block to see the Position vs. Time graph.
   - **Angle Plot:** Double-click the **Angle of Robot** block to see the Angle vs. Time graph.
   - **State-Space Parameters:** Double-click the **Linear Equation Block** to view the generated $A, B, C,$ and $D$ matrices.

### 3. Running Analysis Scripts
1. **PID Visualization:** Run `main2.m` in the MATLAB Command Window to view the PID output stability plots.
2. **Autonomous Simulation:** Run `self_balancing_bot_matlab.m` to observe the robot's pre-programmed forward and backward balancing motion.

## Note on Results
All generated plots from the steps above should match the images provided in the `results/` folder and the data discussed in the `report/report.pdf`.
