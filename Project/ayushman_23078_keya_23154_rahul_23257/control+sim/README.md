# Control & Simulation

This document covers how to build, run, and launch the ROS 2 simulation environment for the DeepONet-MPC waypoint navigation project.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- An X server running on the host (required for Gazebo and RViz display)

---

## Display Access (run on host before anything else)

Gazebo and RViz are GUI applications that need access to your host X display. Run this **once on the host machine** (outside the container) before starting Docker:

```bash
xhost +local:root
```

This grants the container's root user permission to open windows on your display. You should see:

```
non-network local connections being added to access control list
```

To revoke access again after you are done:

```bash
xhost -local:root
```

> **WSL2 users:** Make sure an X server (VcXsrv, Xming, or similar) is running on Windows and that `DISPLAY` is set in your WSL2 environment, for example `export DISPLAY=:0`, before running `xhost`.

---

## Setup

### 1. Build the Docker image

```bash
docker compose build
```

This installs all ROS 2 Jazzy dependencies, the Python environment, and builds the workspace.

### 2. Start the container

```bash
docker compose up -d
```

Starts the container in the background.

### 3. Open a shell inside the container

```bash
docker exec -it ecs334_container bash
```

All subsequent commands are run from inside this shell.

---

## Running the Simulation

The launch file starts Gazebo, RViz, the EKF localisation stack, and the MPC controller in one command. **Gazebo and RViz will open first — wait a few seconds for the simulation to initialise before the robot begins moving.**

There are two waypoint variants corresponding to the two planners evaluated in the paper.

### Option A — Standard A\*

```bash
ros2 launch waypoint_nav combined_sim.launch.py \
    waypoints_file:=/root/ros2_ws/src/waypoint_nav/config/waypoints_A_star.csv
```

### Option B — A\* with Joint Probabilities (ours)

```bash
ros2 launch waypoint_nav combined_sim.launch.py \
    waypoints_file:=/root/ros2_ws/src/waypoint_nav/config/waypoints_joint_prob.csv
```

---

## What Gets Launched

| Component | Description |
|---|---|
| Gazebo | Physics simulation on the rugged terrain world |
| RViz | Live visualisation of path, odometry, and waypoints |
| EKF stack | Local + global localisation fusing wheel odometry, IMU, and GPS |
| NavSat transform | Converts GPS fixes to map-frame odometry |
| MPC controller | DeepONet-MPC with online gain estimation |
| Path publisher | Publishes the waypoint path to `/mpc/waypoints` |

---

## Monitoring

Once running, you can inspect the controller status in a second terminal inside the container:

```bash
# Live MPC status (errors, gains, solver state)
ros2 topic echo /mpc/status

# Estimated actuator gains
ros2 topic echo /mpc/gains

# Wheel commands
ros2 topic echo /mpc/wheel_cmds
```

To record a bag for post-run analysis:

```bash
ros2 bag record /mpc/status /mpc/gains /mpc/wheel_cmds /mpc/errors /odometry/global \
    -o ~/mpc_logs/run_001
```

---

## Troubleshooting

**Gazebo opens but the robot does not move**
The controller waits for a valid odometry message. Wait 5–10 seconds for the EKF to converge. Check with `ros2 topic echo /odometry/global`.

**Display not found / RViz crashes**
Make sure your X server is running and `DISPLAY` is set correctly on the host before starting the container. For WSL2, use an X server such as VcXsrv or set `DISPLAY=:0`.

**`tf_transformations` import error**
Rebuild the image — the Dockerfile must include both `ros-jazzy-tf-transformations` and `python3-transforms3d`.

```bash
docker compose build --no-cache
```