# Functionality: Centralizes all tuning parameters.
#  If you ever want to make the track longer or the lanes wider, you only change it here.
import numpy as np

# Track Geometry Parameters
STRAIGHT_LEN = 100.0  
CURVE_RADIUS = 30.0   
LANE_WIDTH = 4.0      
NUM_LANES = 3

# Derived Track Constants
S1 = STRAIGHT_LEN
S2 = STRAIGHT_LEN + np.pi * CURVE_RADIUS
S3 = 2 * STRAIGHT_LEN + np.pi * CURVE_RADIUS
TRACK_LENGTH = 2 * STRAIGHT_LEN + 2 * np.pi * CURVE_RADIUS


# These weights dictate the "personality" of the controller.
# Simulation Parameters
DT = 0.1  
EGO_SPEED = 15.0 # We use this to find the lap time

# Calculate exactly how many frames 1 lap takes, plus 10 extra frames for a buffer
SIM_FRAMES = int(TRACK_LENGTH / (EGO_SPEED * DT)) + 10


# --- SMPC Parameters ---
N_HORIZON = 20        # Prediction horizon (20 steps * 0.1s = 2.0 seconds ahead)
V_REF = 15.0          # Target speed for the Ego vehicle

# Cost Weights
Q_S = 0.0             # We don't penalize arc length (we just want to move forward)
Q_EY = 50.0           # High penalty for deviating from the lane center
Q_EPSI = 10.0         # Penalty for heading deviation
Q_V = 20.0            # Penalty for deviating from target speed
R_A = 1.0             # Penalty for hard acceleration/braking
R_YAW = 100.0         # Penalty for sharp steering (comfort)

# Actuation Limits
MAX_ACCEL = 3.0       # m/s^2
MIN_ACCEL = -5.0      # m/s^2
MAX_YAW_RATE = 0.5    # rad/s