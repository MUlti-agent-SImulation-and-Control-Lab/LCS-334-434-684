# UGV-UAV Multi-Agent Control & Perception: Observability-Aware Mobile Trailing

This repository contains the complete implementation for the Phase 2 3D Event-Triggered Tube MPC project. A UAV autonomously tracks a UGV while maximizing observability and minimizing uncertainty.

## 📂 Project Structure

```text
Project_ws/
├── src/
│   └── ugv_uav_control/
│       ├── launch/
│       │   └── middle_path_follower.launch.py   # Main launch file for all conditions
│       ├── models/                              # Custom Gazebo models (x500 UAV, TurtleBot3 UGV, Track)
│       ├── ugv_uav_control/                     # Core ROS 2 Python Nodes
│       │   ├── middle_path_follower.py          # Vision & Perception (ArUco + Path extraction)
│       │   ├── ekf_node.py                      # Extended Kalman Filter for UGV state
│       │   ├── belief_mpc_node.py               # Belief-Space MPC for UGV control
│       │   ├── uav_mpc_node.py                  # Event-Triggered 3D MPC for UAV optimal trailing
│       │   ├── uav_ancillary_node.py            # High-frequency LQR Tube Controller
│       │   ├── ugv_model.py                     # Shared kinematics and Jacobians
│       │   ├── view_quality.py                  # Differentiable View Quality metric (q_k)
│       │   └── experiment_logger.py             # CSV Logging for performance metrics
│       ├── scripts/
│       │   └── evaluate_results.py              # Data analysis and plotting script
│       ├── setup.py                             # Python package configuration
│       └── package.xml                          # ROS 2 package dependencies
├── docs/                                        # Extended documentation and math formulations
└── README.md                                    # This file
```

## ⚙️ Dependencies & Installation

This project is built for **ROS 2 (Humble/Iron/Jazzy)** and **Gazebo Sim**.

1. **Clone the repository** (or extract the submission archive) into your workspace.
2. **Install dependencies**:
   ```bash
   sudo apt install ros-humble-ros-gz ros-humble-cv-bridge
   pip install numpy scipy pandas matplotlib opencv-python
   ```
   *(Note: Replace `humble` with your ROS 2 distro if using Iron or Jazzy).*

3. **Build the workspace**:
   ```bash
   cd ~/Desktop/Project_ws  # Navigate to the workspace root
   colcon build --packages-select ugv_uav_control
   source install/setup.bash
   ```

## 🚀 Execution Instructions

The code is strictly designed to run out-of-the-box without any modifications. We have provided two distinct launch configurations corresponding to the experiments detailed in the final report.

### Condition 1: Phase 1 Baseline (1Hz Fixed Rate, No Tube)
To run the legacy static-rate baseline for comparison:
```bash
ros2 launch ugv_uav_control middle_path_follower.launch.py controller:=belief_mpc use_timer:=true use_ancillary:=false duration:=60.0 experiment_name:=phase1_baseline
```

### Condition 2: Phase 2 Full System (Event-Triggered Tube MPC)
To run the full observability-aware architecture (Recommended):
```bash
ros2 launch ugv_uav_control middle_path_follower.launch.py controller:=belief_mpc use_timer:=false use_ancillary:=true duration:=60.0 experiment_name:=phase2_full
```

*Note: A Gazebo GUI window and a CV2 debug window (showing the heatmap and ArUco detection) will open automatically. The simulation runs for 60 seconds and gracefully exits.*

## 📊 Expected Results

Upon completion of either launch command, the `experiment_logger` node will automatically save a `.csv` log file and a `_summary.txt` file in the `~/experiment_logs/` directory.

As detailed in our final report, evaluating the logs against the two conditions should yield the following approximate metrics (excluding the first 10s of transient startup):

| Metric | Phase 1 Baseline | Phase 2 Full System | Improvement |
|--------|------------------|---------------------|-------------|
| **Mean Tracking Error** | ~1.037 m | **~0.463 m** | **~55%** |
| **ArUco Visibility** | ~46.3% | **~87.0%** | **~88%** |
| **ArUco Dropouts** | 55 | **7** | **~87%** |
| **MPC Solve Time** | ~50.0 ms | **~16.0 ms** | **3.1x Faster** |

The results mathematically prove that the Phase 2 Event-Triggered Tube MPC architecture significantly outperforms the baseline, safely bounding tracking errors while maintaining optimal view geometry within a strict 150ms real-time compute budget.
