# Bipedal Walking Robot using Model Predictive Control (MPC)

**Authors:** SUDESHNA RAY (Roll-23334), AVANI BARANWAL (Roll-23047), RAMANDEEP SINGH HORA (Roll-23265)  
**Course:** ECS-334/434/684 (Linear Control Systems)

## Project Overview
This project simulates the walking gait of a bipedal robot using a Model Predictive Controller (MPC). The physical robot kinematics and dynamics are modeled using Simscape Multibody, while the MPC algorithm generates the optimal control actions to maintain balance and achieve forward locomotion.

## Repository Navigation
This folder (`Project/sudeshna_23334/`) is organized as per the course submission guidelines:

* **`src/`**: Contains the source codes (Simulink models and code files).
  * `bipedal_walk_sims` - Folder containing the 2 simulations as follows:
  * `ContactForce.slx` - The Simulink/Simscape Multibody model which establishes the contact force between the robo feet and the ground solid. Without this file, the robot falls through the ground.
  * `BipedalWalk.slx` - The main Simulink/Simscape Multibody model: Demonstrates the bipedal walking of the robot.
  * `StateSpace.m` - MATLAB script that defines the 2nd order LIPM model (using ZMP and COM difference) as a 1st order state-space representation, defines the control horizon limits and other constraints for the MPC.
  * `ref.m` - The script functions as the Trajectory Generator. By generating a discrete ZMP staircase for the sagittal plane and a periodic oscillation for the lateral plane, it provides the necessary setpoints for the controller to execute a stable, dynamic gait while maintaining the ZMP within the robot's foot in each successive footstep.
  * `GaitScheduler.m` - It generates the joint-space trajectories/kineamaics for the leg movement. It implements the finite-state logic required to transition each leg between 'Swing' and 'Stance' phases, ensuring that while the MPC maintains balance, the physical limbs execute the lifting and placement necessary to achieve the desired step length and height.
  * `robotTree.m` - The MATLAB script initializes a RigidBodyTree object that serves as the kinematic model for the biped. While the Simscape Multibody model represents the Physical Plant (the 'hardware' being simulated), the robotTree.m provides the Kinematic and Dynamic Model required by the control algorithms. It eases Jacobian and other calculations.
  
* **`logs/`**: Contains the logged `.mat` data files (workspace variables, scope data) generated after running the simulation.

* **`results/`**: Contains the output visualizations.
  * `contactforce_Sim.mov` - Video recording of the Simscape 3D mechanics explorer, showing the drop-test result of how the robot falls from a height and stops at the ground, showing contact force is established.
  * `BipedalWalk_sim.mov` - Video recording of the Simscape 3D mechanics explorer, shows the bipedal walk for 10 seconds.
  * `robotSIM` - Image of the robot model simulated using Simscape Multibody.
  * `COMwithoutMPC` - PS-Simulink converter data analysis scope output data, showing where COM of the the standing robot is situated, this COM value is essential as an input to the MPC.
  * `COMwithMPC` - PS-Simulink converter data analysis scope output data, showing where COM of the the walking robot is situated, the positive values between 0.88m to 0.92m (2cm deviation) height for the whole simulation duration shows the robot has almost stable walking COM and it does not fall.
  * `Output_stateSpace` - Output of StateSpace.m
  * `Output_RobotTree` - Output of robotTree.m
  * `Output_Gaitschedulerandref` - Output of GaitScheduler.m and ref.m
  * `jointreferencetrajectoriesoutput` - The image is a snapshot of the Reference Trajectories for a bipedal robot. It tracks four critical degrees of freedom (DOF) for both the Left (L) and Right (R) legs: Hip pitch, Knee, Ankle Pitch and Hip Roll. These plots represent the "ideal" path or the goal that the controller is trying to track.
  * `robotbodywithoutMPC and robotlegwithoutMPC` - Simscape multibody blocks image of the primary robot body without MPC imprementation.
  * `robotbodywithMPC and robotlegwithMPC` - Simscape multibody blocks image of the robot body with MPC imprementation.
  * `MPC_MV_output_rightlegleg and MPC_MV_output_leftleg` - Scope output of the manipulated Variables of MPC (ZMP velocity of right leg and left leg respectively).
  
* **`report/`**: Contains `report.pdf`, detailing the mathematical modeling, controller design, and analysis of the results.

* **`requirements.txt`**: Lists all required MATLAB toolboxes.

## Step-by-Step Instructions to Run the Project

Please follow these exact steps to execute the code and reproduce the results discussed in the report:

**STEP 1 – Build the robot tree**
1. open MATLAB.
2. run robotTree.m      →  creates 'robot' in workspace.

**STEP 2 – Build MPC controller**
1. run StateSpace.m     →  creates 'mpc_controller' and 'x0' in workspace.

**STEP 3 – Build ZMP walking reference**
1. run ref.m            →  creates 'ref_ts' timeseries (4 outputs) in workspace.

**STEP 4 – Build joint angle gait**
1. run GaitScheduler.m  →  creates 'joint_ref' timeseries (8 joints) in workspace.

**STEP 5 - Establish contact force between World Ground and robot feet**
1. open SIMULINK/SIMSCAPE MULTIBODY
2. run the ContactForce.slx simulink model  →  shows how robot falls from a height and stops on the ground, bending due to impact.

**STEP 6 - Final robot gait generation**
1. run the BipedalWalk.slx simulink model  →  the robot walks without falling (COM always above ground).

# Note: (Very Important!) Simscape results can vary based on different systems, computers, or model configurations. While a model should theoretically produce the same results, differences often arise due to solver settings, numerical tolerances, or software version differences.
