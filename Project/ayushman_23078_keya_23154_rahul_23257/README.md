# Terrain-Aware Navigation for Husky A200
### Fault-Tolerant MPC via DeepONet and EKF Sensor Fusion

> ECS334 Linear Control Systems — Group Project

**Ayushman Saha · Keya Sinha · Rahul Kiran Jana**

---

## Overview

This project implements a fault-tolerant autonomous navigation system for the Clearpath Husky A200 differential-drive robot operating on uneven, rugged outdoor terrain. The system combines:

- **A\* path planning with joint probability terrain cost** to generate safe waypoint sequences that avoid hazardous terrain features
- **DeepONet-based MPC** for real-time trajectory tracking, using a neural operator to predict robot dynamics
- **Online gain estimation** via an EKF sensor fusion stack to detect and compensate for actuator faults and terrain-induced slip in real time

---

## Repository Structure

```
.
├── control_sim/        # MPC controller + Gazebo/RViz simulation (ROS 2 Jazzy)
│   └── README.md
│
└── path_planner/       # A* planner with joint probability terrain cost
    └── README.md
```

### `control_sim/`
Contains the full ROS 2 workspace with the DeepONet-MPC controller node, EKF localisation stack, Gazebo simulation launch files, and Docker environment. See [`control_sim/README.md`](./control_sim/README.md) for setup and usage instructions.

### `path_planner/`
Contains the terrain-aware A\* planning algorithm with joint probability cost maps. Outputs waypoint CSV files consumed by the controller. See [`path_planner/README.md`](./path_planner/README.md) for setup and usage instructions.

---

## Quickstart

For full setup instructions refer to the README in each subfolder. The high-level flow is:

1. Run the path planner to generate a waypoints CSV
2. Build and launch the simulation with the generated CSV
3. Observe the robot navigate the terrain in Gazebo and RViz

---