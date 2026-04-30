# config.py

# Simulation parameters
DT = 0.1  # Time step in seconds
SIM_TIME = 10.0  # Total simulation time per episode

# Road geometry
LANES = [-3.5, 0.0, 3.5]  # y-coordinates of the 3 lanes

# Vehicle parameters
# Collision Ellipse Parameters (Ego Circle + Target Ellipse)
ELLIPSE_A = 3.7  # Longitudinal semi-axis (meters)
ELLIPSE_B = 2.2  # Lateral semi-axis (meters)

COLLISION_RADIUS = 2.5  # Euclidean distance to trigger collision
MAX_VELOCITY = 30.0  # m/s
MIN_VELOCITY = 0.0   # m/s

# Reward Function Weights
W_PROGRESS = 1.0     # Reward for moving forward (velocity)
W_LANE = -0.5        # Penalty for deviating from the center of a lane
W_COLLISION = -100.0 # Heavy penalty for crashing