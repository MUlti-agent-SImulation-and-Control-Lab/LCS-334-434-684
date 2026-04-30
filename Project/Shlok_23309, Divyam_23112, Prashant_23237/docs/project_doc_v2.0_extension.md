# Phase 2: Observability-Aware Mobile Trailing

This phase extends the deterministic v1.1 system by introducing a mobile UAV platform and a belief-space optimizer. The primary objective is to maintain high-quality viewing geometry over a ground target while minimizing estimation uncertainty during non-linear maneuvers.

## 6. Architectural Evolution from v1.1

The v2.0 upgrade transitions the UAV from a static 10m overhead camera to a mobile 2.5D trailer (2D motion at a fixed 5m altitude). This shift introduces several key architectural enhancements:

1.  **Mobile Sensor Platform**: The UAV platform accepts $v_x, v_y$ commands to track the UGV. The lower altitude (5m) improves pixel density on target but reduces the ground field-of-view (FoV) radius to $r_{fov} \approx 2.89$m.
2.  **Unified Dynamics Node (`ugv_model.py`)**: To eliminate logic duplication, all kinematic process models and EKF Jacobians are consolidated into a shared utility. This ensures that the UGV's internal MPC, the EKF, and the UAV's belief-space optimizer utilize identical physical assumptions.
3.  **Predictive Trajectory Awareness**: The UAV node now subscribes to `/ugv_predicted_traj` (published by the UGV MPC). This allows the UAV to "anticipate" UGV path turns before they occur, using a hybrid extrapolation strategy (Path Interpolation + Constant Velocity Projection).

---

## 7. Mathematical Formulation of Dynamic Observability

### 7.1 Differentiable View Quality ($q_k$)
To optimize the UAV trajectory, we define a view quality metric $q(d) \in (0, 1)$ that is continuous and differentiable. Unlike hard FoV constraints, this allows the gradient-based optimizer to move the UAV toward the UGV even when the target is temporarily lost.

The metric is composed of two factors:
1.  **Radial Sigmoid ($q_{radial}$)**: Penalizes horizontal distance $d$ relative to the FoV radius $r_{fov}$:
    $$q_{radial}(d) = \frac{1}{1 + \exp(\kappa \cdot (d - r_{fov}))}$$
    Where $\kappa=5.0$ controls the roll-off steepness.

2.  **Centering Bonus ($q_{center}$)**: Rewards keeping the target near the optical axis using a Gaussian envelope:
    $$q_{center}(d) = \exp\left(-\frac{d^2}{2\sigma^2}\right)$$
    Where $\sigma = \frac{r_{fov}}{2}$.

The final view quality is the product: $q = q_{radial} \cdot q_{center}$.

### 7.2 Dynamic Measurement Noise ($R_k$)
In v1.1, the ArUco measurement noise $R_{base}$ was static. In v2.0, $R_k$ is linked directly to the view quality. As the target moves toward the FoV edge, $q_k$ decays, effectively increasing the measurement covariance and forcing the EKF to rely more on the process model:
$$R_k(q) = \frac{1}{q_k} R_{base}$$

---

## 8. Receding Horizon Belief-Space Optimization

The UAV plans its velocity commands by minimizing an objective function that accounts for future belief states.

### 8.1 Objective Function ($J_{UAV}$)
The optimizer solves for a sequence of velocity commands $\mathbf{u} = [u_1, \dots, u_N]$ over $N=8$ steps ($\Delta t = 0.3$s) to minimize:
$$J = \sum_{k=1}^N \left[ \lambda \cdot \text{tr}(\Sigma_{k|k}) + \gamma_q \cdot (1 - q_k) + \beta \cdot \|u_k\|^2 \right]$$
Where:
- $\text{tr}(\Sigma_{k|k})$: Trace of the predicted UGV covariance (Uncertainty penalty).
- $(1 - q_k)$: Reward for centering (Observability penalty).
- $\|u_k\|^2$: Effort penalty to prevent jitter.

### 8.2 Sequential Belief Propagation
At each horizon step $k$, the UAV predicts the UGV's covariance using the Extended Kalman Filter (EKF) equations:
1.  **Predict**: $\Sigma_{k+1|k} = F_k \Sigma_{k|k} F_k^\top + Q$
2.  **Update**: $\Sigma_{k+1|k+1} = (I - K_{k+1} H) \Sigma_{k+1|k}$
    - $K_{k+1} = \Sigma_{k+1|k} H^\top (H \Sigma_{k+1|k} H^\top + R_{k+1})^{-1}$
    - $R_{k+1}$ is computed using the $q_{k+1}$ resulting from the planned UAV move $u_{k+1}$.

---

## 9. Comparative Evaluation and Results

Final 180-second verification runs were conducted to compare the Belief-Space MPC against a reactive Centroid-Following baseline.

| Metric | Centroid Baseline | Full Belief-Space MPC | v1.1 Static (10m) |
| :--- | :--- | :--- | :--- |
| **Visibility Rate** | **98.06%** | 93.53% | 98.00% |
| **Mean tr(Sigma)** | 0.0557 | **0.0491** | 0.008 |
| **UGV Tracking Err** | 1.168m | **1.064m** | 0.120m |
| **Mean q_k** | 0.801 | **0.849** | 0.850 |
| **Mean Separation** | 0.791m | **0.656m** | 10.00m |

### 9.1 The "Visibility Paradox" Analysis
While the MPC achieves significantly lower uncertainty and better view quality, its raw visibility (93.5%) is slightly lower than the sluggish baseline (98.1%). 
**Mechanistic Explanation**: The baseline stays conservatively centered at low speeds. The MPC is more aggressive ($v_{max}=1.5$m/s) to optimize viewing angles during corners. At 5m altitude, the tight FoV (2.89m radius) makes the system hypersensitive to high-speed overshoots, resulting in brief FoV exits that are a geometric trade-off rather than a control failure.

---

## 10. Implementation Details

### 10.1 Key ROS Parameters
Nodes find their weights in the `ugv_uav_control` namespace:
- `lambda`: `1.0` (Uncertainty weight)
- `gamma_q`: `1.5` (View quality reward)
- `beta`: `0.1` (Effort weight)
- `force_fallback`: `False` (Force centroid-following mode for study)

### 10.2 System Limitations
- **Geometric FoV Constraints**: High-speed maneuvers at 5m altitude are inherently risky due to the small ground footprint.
- **Predictive Gaps**: Beyond the 1.0s UGV MPC horizon, the UAV relies on constant-velocity extrapolation which may diverge during sudden UGV stops.
- **Perfect UAV assumption**: UAV localization is assumed to have zero noise; cross-covariance between platforms is not currently tracked.

---

## 11. Deployment Instructions

### Launching the Full Phase 2 System (Tube MPC)
```bash
ros2 launch ugv_uav_control middle_path_follower.launch.py controller:=belief_mpc use_timer:=false use_ancillary:=true
```

### Running Evaluation Conditions
To toggle between the full Phase 2 optimizer and the Phase 1 simple baseline:
```bash
# Full Phase 2 Tube MPC (Event-Triggered + Ancillary)
ros2 launch ugv_uav_control middle_path_follower.launch.py controller:=belief_mpc use_timer:=false use_ancillary:=true

# Phase 1 Baseline Only (1Hz Fixed Rate, No Ancillary)
ros2 launch ugv_uav_control middle_path_follower.launch.py controller:=belief_mpc use_timer:=true use_ancillary:=false
```
