"""
View Quality Module
===================
Computes a scalar view quality score q ∈ [0, 1] given UAV (x, y, z) and
UGV (x, y) positions.

q is maximized when:
1. UGV is directly below the UAV (centered in FoV)
2. UAV altitude is near z_opt (optimal resolution/precision)

The function is **fully differentiable** and suitable for gradient-based 
optimization (SLSQP, etc.).
"""

import numpy as np


def compute_view_quality(
    uav_x: float,
    uav_y: float,
    uav_z: float,
    ugv_x: float,
    ugv_y: float,
    fov_angle: float = 1.047,
    sharpness: float = 5.0,
    z_opt: float = 5.0,
    sigma_z: float = 3.0,
) -> float:
    """
    Compute a smooth, differentiable 3D view quality score.

    Parameters
    ----------
    uav_x, uav_y, uav_z : float
        UAV position (world frame).
    ugv_x, ugv_y : float
        UGV position (world frame).
    fov_angle : float
        Full camera FoV angle (rad). Default: 1.047 (60°).
    sharpness : float
        Sigmoid steepness (κ) for the boundary roll-off.
    z_opt : float
        Optimal altitude (m) for sensor precision.
    sigma_z : float
        Standard deviation for altitude-based quality penalty.

    Returns
    -------
    q : float
        View quality score in (0, 1).
    """
    # 0. Derived geometry
    # FoV radius increases linearly with altitude: r = z * tan(theta/2)
    r_fov = uav_z * np.tan(fov_angle / 2.0)
    # Ensure r_fov is positive (though z should be >= 1.0 in practice)
    r_fov = np.maximum(r_fov, 0.1)

    # 1. Horizontal distance
    dx = uav_x - ugv_x
    dy = uav_y - ugv_y
    d = np.sqrt(dx * dx + dy * dy + 1e-12)

    # 2. Radial sigmoid (FoV boundary visibility)
    # Transitions from 1 (inside) to 0 (outside)
    q_radial = 1.0 / (1.0 + np.exp(sharpness * (d - r_fov)))

    # 3. Centering bonus (Resolution/Perspective)
    # Rewards keeping the target near the optical axis
    sigma_xy = r_fov / 2.0
    q_center = np.exp(-(d * d) / (2.0 * sigma_xy * sigma_xy))

    # 4. Altitude precision scale s(z)
    # Gaussian penalty centered at z_opt
    s_z = np.exp(-(uav_z - z_opt)**2 / (2.0 * sigma_z * sigma_z))

    return float(q_radial * q_center * s_z)


def compute_view_quality_gradient(
    uav_x: float,
    uav_y: float,
    uav_z: float,
    ugv_x: float,
    ugv_y: float,
    **kwargs,
) -> tuple:
    """
    Compute numerical gradient of view quality w.r.t. UAV position (x, y, z).

    Returns (dq/dx, dq/dy, dq/dz) via central differences.
    """
    eps = 1e-6
    dq_dx = (
        compute_view_quality(uav_x + eps, uav_y, uav_z, ugv_x, ugv_y, **kwargs)
        - compute_view_quality(uav_x - eps, uav_y, uav_z, ugv_x, ugv_y, **kwargs)
    ) / (2.0 * eps)
    dq_dy = (
        compute_view_quality(uav_x, uav_y + eps, uav_z, ugv_x, ugv_y, **kwargs)
        - compute_view_quality(uav_x, uav_y - eps, uav_z, ugv_x, ugv_y, **kwargs)
    ) / (2.0 * eps)
    dq_dz = (
        compute_view_quality(uav_x, uav_y, uav_z + eps, ugv_x, ugv_y, **kwargs)
        - compute_view_quality(uav_x, uav_y, uav_z - eps, ugv_x, ugv_y, **kwargs)
    ) / (2.0 * eps)
    return dq_dx, dq_dy, dq_dz


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ugv = (0.0, 0.0)
    
    # --- Test A: Radial sweep at fixed z=5.0 ---
    offsets = np.linspace(0.0, 6.0, 200)
    scores_radial = [compute_view_quality(d, 0.0, 5.0, *ugv) for d in offsets]
    
    # --- Test B: Altitude sweep at fixed offset=0.0 ---
    altitudes = np.linspace(1.0, 12.0, 200)
    scores_alt = [compute_view_quality(0.0, 0.0, z, *ugv) for z in altitudes]
    grads_alt = [compute_view_quality_gradient(0.0, 0.0, z, *ugv)[2] for z in altitudes]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Radial Plot
    ax1.plot(offsets, scores_radial, 'b-', label='q(d) at z=5.0')
    ax1.axvline(5.0 * np.tan(1.047/2), color='r', linestyle='--', label='r_fov at 5m')
    ax1.set_xlabel('Horizontal Offset (m)')
    ax1.set_ylabel('Quality q')
    ax1.set_title('View Quality vs. Horizontal Offset')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Altitude Plot
    ax2.plot(altitudes, scores_alt, 'g-', label='q(z) centered')
    ax2.plot(altitudes, grads_alt, 'm--', label='dq/dz centered')
    ax2.axvline(5.0, color='k', linestyle=':', label='z_opt = 5.0')
    ax2.set_xlabel('Altitude z (m)')
    ax2.set_ylabel('Quality / Gradient')
    ax2.set_title('View Quality vs. Altitude (Predictive Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = '/home/shlok-mehndiratta/Desktop/Project_ws/view_quality_3d_test.png'
    plt.savefig(out_path, dpi=150)
    print(f'Test plots saved to {out_path}')

    # --- Verification ---
    print('\n=== 3D View Quality Verification ===')
    q_opt = compute_view_quality(0,0, 5, 0,0)
    q_low = compute_view_quality(0,0, 2, 0,0)
    q_high = compute_view_quality(0,0, 10, 0,0)
    
    print(f'q at z_opt (5.0m): {q_opt:.4f}')
    print(f'q at z_low (2.0m): {q_low:.4f}')
    print(f'q at z_high (10.0m): {q_high:.4f}')
    
    assert q_opt > q_low and q_opt > q_high, "Altitude peak not at z_opt"
    print("\n✅ Step 1 Verification Complete: Altitude peaks correctly and decays smoothly.")
