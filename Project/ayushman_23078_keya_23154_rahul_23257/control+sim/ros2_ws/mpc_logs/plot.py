import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
df = pd.read_csv('/root/ros2_ws/mpc_logs/plots/master_data.csv')
wps = pd.read_csv('/root/ros2_ws/src/waypoint_nav/config/waypoints.csv') # Assuming columns are x, y, z

# Create a folder for the final report
output_dir = 'presentation_charts'
os.makedirs(output_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-paper') # Professional scientific style
params = {'legend.fontsize': 'x-large', 'axes.labelsize': 'x-large', 
          'axes.titlesize': 'xx-large', 'xtick.labelsize': 'large', 'ytick.labelsize': 'large'}
plt.rcParams.update(params)

# ---------------------------------------------------------
# 1. PATH TRACKING (XY Trajectory)
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))

plt.plot(df['x'], df['y'], 'b-', lw=2, label='Robot Path', zorder=2)
sc = plt.scatter(
    wps['x'], wps['y'],
    c=wps['z'], cmap='terrain',
    norm=plt.Normalize(vmin=wps['z'].min(), vmax=wps['z'].max()),
    s=80, linewidths=1.0, edgecolors='black', zorder=3, label='Waypoints'
)

plt.title("Spatial Tracking: Actual Path vs 3D Waypoints")
plt.xlabel("X [m]"); plt.ylabel("Y [m]")
plt.axis('equal')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.savefig(f'{output_dir}/01_path_tracking_3d_wps.png', dpi=300)
# ---------------------------------------------------------
# 2. ERROR CONVERGENCE (The "Performance" Proof)
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

ax1.plot(df['t_rel'], df['dist_err'], 'g-', label='Distance Error [m]', lw=2)
ax2.plot(df['t_rel'], df['head_err'], 'r--', label='Heading Error [deg]', alpha=0.6)

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Distance Error [m]', color='g')
ax2.set_ylabel('Heading Error [deg]', color='r')
plt.title("Control Error Convergence")
plt.grid(True, which='both', axis='x', linestyle='--')
plt.savefig(f'{output_dir}/02_errors.png', dpi=300)

# ---------------------------------------------------------
# 3. FAULT TOLERANCE (The "Intelligence" Proof)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df['t_rel'], df['gL'], label='Left Wheel Gain (Efficiency)', color='blue')
plt.plot(df['t_rel'], df['gR'], label='Right Wheel Gain (Efficiency)', color='orange')
plt.axhline(y=1.0, color='black', linestyle=':', label='Nominal (Healthy)')
plt.fill_between(df['t_rel'], df['gL'], 1.0, color='blue', alpha=0.1)
plt.title("Online Gain Estimation (Slip/Fault Detection)")
plt.xlabel("Time [s]")
plt.ylabel("Gain Factor")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/03_fault_tolerance.png', dpi=300)

# ---------------------------------------------------------
# 4. VELOCITY TRACKING (The "Performance" Proof)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df['t_rel'], df['v_l'], label='Left Wheel Velocity', alpha=0.7, color='blue')
plt.plot(df['t_rel'], df['v_r'], label='Right Wheel Velocity', alpha=0.7, color='orange')
plt.fill_between(df['t_rel'], df['v_l'], 0.0, color='blue', alpha=0.1)
plt.fill_between(df['t_rel'], df['v_r'], 0.0, color='orange', alpha=0.1)
plt.title("Velocity inputs")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.savefig(f'{output_dir}/04_velocity_tracking.png', dpi=300)


print(f"Finished! All meaningful charts are in the '{output_dir}' folder.")