<<<<<<< HEAD
# Autonomous Driving with Stochastic MPC (SMPC)

## Project Overview
This is a research-oriented simulation project focused on autonomous driving decision-making. The goal is to simulate an Ego Vehicle (EV) performing safe lane changes on a 3-lane highway in the presence of target vehicles. 

Ultimately, this project will implement a **Stochastic Model Predictive Control (SMPC)** framework to handle the uncertain, multimodal future behaviors of surrounding vehicles (e.g., probability of keeping a lane vs. changing a lane).

## Current Status: Phase 1 - Environment Validation
We have successfully built a vectorized, 2D kinematic simulation environment and an interactive spatio-temporal visualizer. 

### Environment Details
* **Road Geometry:** 3 lanes with fixed lateral positions at `y = [-3.5, 0.0, 3.5]` meters.
* **Time:** Discrete simulation with `dt = 0.1` seconds.
* **State Space:** Each vehicle is tracked via `[x, y, velocity]`.
* **Control Inputs:** `[longitudinal_acceleration, lane_change_rate]`.
* **Dynamics:** Simplified explicit Euler kinematic model.

### Collision Model (The Minkowski Sum)
To realistically model lane-passing without triggering false collisions, we use a hybrid collision boundary. The Ego Vehicle is modeled as a circle, and Target Vehicles are modeled as ellipses (longer longitudinally, narrower laterally). 

Mathematically, we simplify this by treating the Ego Vehicle as a single point and expanding the Target Vehicle into an "Effective Ellipse". A collision is detected if the Ego Vehicle enters this boundary:

$$\frac{(x_{ego} - x_{target})^2}{A^2} + \frac{(y_{ego} - y_{target})^2}{B^2} \le 1$$

Where `A` and `B` are the combined semi-axes representing our longitudinal and lateral safety margins.

## Project Structure
The project is kept modular for easy iteration from basic control to advanced SMPC.

* `config.py`: Centralized physical constants, reward weights, and safety boundaries.
* `dynamics.py`: 2D kinematic vehicle model and state updates.
* `env.py`: The simulation orchestrator handling state transitions, rewards, and collision checks.
* `plot.py`: Interactive Matplotlib visualizer with a time-slider to debug spatio-temporal trajectories.
* `run.py`: Execution script. Currently runs open-loop dummy actions to validate physics.
* `mpc_controller.py`: *(Upcoming)* Deterministic MPC using CVXPY.
* `smpc_controller.py`: *(Upcoming)* Stochastic MPC handling multimodal predictions.

## How to Run
Ensure you have `numpy` and `matplotlib` installed. Run the simulation test script via your terminal:

```bash
=======
# Autonomous Driving with Stochastic MPC (SMPC)

## Project Overview
This is a research-oriented simulation project focused on autonomous driving decision-making. The goal is to simulate an Ego Vehicle (EV) performing safe lane changes on a 3-lane highway in the presence of target vehicles. 

Ultimately, this project will implement a **Stochastic Model Predictive Control (SMPC)** framework to handle the uncertain, multimodal future behaviors of surrounding vehicles (e.g., probability of keeping a lane vs. changing a lane).

## Current Status: Phase 1 - Environment Validation
We have successfully built a vectorized, 2D kinematic simulation environment and an interactive spatio-temporal visualizer. 

### Environment Details
* **Road Geometry:** 3 lanes with fixed lateral positions at `y = [-3.5, 0.0, 3.5]` meters.
* **Time:** Discrete simulation with `dt = 0.1` seconds.
* **State Space:** Each vehicle is tracked via `[x, y, velocity]`.
* **Control Inputs:** `[longitudinal_acceleration, lane_change_rate]`.
* **Dynamics:** Simplified explicit Euler kinematic model.

### Collision Model (The Minkowski Sum)
To realistically model lane-passing without triggering false collisions, we use a hybrid collision boundary. The Ego Vehicle is modeled as a circle, and Target Vehicles are modeled as ellipses (longer longitudinally, narrower laterally). 

Mathematically, we simplify this by treating the Ego Vehicle as a single point and expanding the Target Vehicle into an "Effective Ellipse". A collision is detected if the Ego Vehicle enters this boundary:

$$\frac{(x_{ego} - x_{target})^2}{A^2} + \frac{(y_{ego} - y_{target})^2}{B^2} \le 1$$

Where `A` and `B` are the combined semi-axes representing our longitudinal and lateral safety margins.

## Project Structure
The project is kept modular for easy iteration from basic control to advanced SMPC.

* `config.py`: Centralized physical constants, reward weights, and safety boundaries.
* `dynamics.py`: 2D kinematic vehicle model and state updates.
* `env.py`: The simulation orchestrator handling state transitions, rewards, and collision checks.
* `plot.py`: Interactive Matplotlib visualizer with a time-slider to debug spatio-temporal trajectories.
* `run.py`: Execution script. Currently runs open-loop dummy actions to validate physics.
* `mpc_controller.py`: *(Upcoming)* Deterministic MPC using CVXPY.
* `smpc_controller.py`: *(Upcoming)* Stochastic MPC handling multimodal predictions.

## How to Run
Ensure you have `numpy` and `matplotlib` installed. Run the simulation test script via your terminal:

```bash
>>>>>>> a0716a10d0730f25f94851fde7115e1102c0813c
python run.py