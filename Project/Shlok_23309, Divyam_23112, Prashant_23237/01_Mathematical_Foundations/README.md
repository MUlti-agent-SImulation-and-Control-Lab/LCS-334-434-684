# LCS: TEMPC

This repository contains Model Predictive Control (MPC) packages including Standard MPC, Full Tube MPC, and Event-Triggered (ET) Tube MPC for robust trajectory tracking. It supports both standalone Python simulations and basic ROS2 Humble/Gazebo integration.

---

## Python Simulations

The `simulations/` directory contains all Python standalone implementations of the control suite.

### Requirements
Install the required python packages from [requirements.txt](file:///home/ptc/ros2_ws/src/lcs/requirements.txt) to avoid any version conflicts. 
While every effort has been ensured to list all relevant python packages in requirements.txt, some dependencies might be missing. Please see pip_requirements.txt for an exhaustive list of packages installed in the system.
It is possible to feed the requirements file directly to pip using the `-r` flag:
```bash
pip install -r requirements.txt
```


### File Breakdown

- **[tube_calculation.py](file:///home/ptc/ros2_ws/src/lcs/simulations/tube_calculation.py)**: Performs offline calculations to determine the Robust Positional Invariant (RPI) set (the "Tube"), computes the tracking feedback gain $K$, tightens state and input constraints, and saves all data to `tube_data.npz`.
- **[sim_stress_tests.py](file:///home/ptc/ros2_ws/src/lcs/simulations/sim_stress_tests.py)**: The main stress testing benchmark suite. Compares all controllers across multiple scenarios (e.g., crosswinds, mass/thrust mismatches, sensor glitches) and outputs trajectories, error analysis, and optimization trigger timelines to the `plots/` subdirectory.
- **[sim_et_tube.py](file:///home/ptc/ros2_ws/src/lcs/simulations/sim_et_tube.py)**: Standalone simulation script for Event-Triggered Tube MPC.
- **[sim_full_tube.py](file:///home/ptc/ros2_ws/src/lcs/simulations/sim_full_tube.py)**: Standalone simulation script for Full Tube MPC.
- **[sim_standard_mpc.py](file:///home/ptc/ros2_ws/src/lcs/simulations/sim_standard_mpc.py)**: Standalone simulation script for Standard MPC.

### Running Simulations
1. **Using Defaults**: The project includes pre-calculated RPI set metrics in `tube_data.npz`. You can run the stress testing suite immediately:
   ```bash
   cd simulations/
   python3 sim_stress_tests.py
   ```

The resulting plots and other simulation outputs are saved in the `plots/` subdirectory.

2. **Custom Paths & Parametrization**: If you modify the parameters in `tube_calculation.py`, run the following to regenerate `tube_data.npz` before running simulations:
   ```bash
   python3 simulations/tube_calculation.py
   ```

---

## ROS2 & Gazebo Simulation Integration

The entire workspace can be compiled and run as an `ament_python` ROS2 package.

### Prerequisites & Dependencies
- **OS**: Ubuntu 22.04 LTS
- **ROS Version**: ROS2 Humble Hawksbill
- **Gazebo**: Gazebo Classic 11
- **ROS Packages**:
  - `rclpy`
  - `geometry_msgs`
  - `nav_msgs`
  - `std_msgs`
  - `gazebo_ros_pkgs` 

### Building the Package
To build the ROS2 node and its launch files:
```bash
colcon build --packages-select lcs
source install/setup.bash
```

### Running Nodes via Launch Files

To run the ROS2 simulations correctly, you must launch Gazebo and the waypoint publisher before running any controller node:

1. **Launch Gazebo Environment**:
   ```bash
   ros2 launch lcs simple_gazebo.launch.py
   ```

2. **Run Waypoint Publisher**:
   ```bash
   ros2 launch lcs waypoint_publisher.launch.py
   ```

3. **Run the Controller Node**:
   - **Standard MPC**: `ros2 launch lcs standard_mpc.launch.py`
   - **Tube MPC**: `ros2 launch lcs tube_mpc.launch.py`
   - **ET Tube MPC**: `ros2 launch lcs et_tube_mpc.launch.py`

### Limitations
- **UAV Representation**: We use a simple box instead of a proper drone URDF, as more comprehensive simulations have been prepared elsewhere.
- **Quadcopter Constraints**: The crazyflie related nodes are not functioning in this implementation. For the crazyflie implementation, please refer to the other part of our project where we have implemented the Software in the Loop (SITL).
