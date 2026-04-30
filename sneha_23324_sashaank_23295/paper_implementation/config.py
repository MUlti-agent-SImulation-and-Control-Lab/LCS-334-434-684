<<<<<<< HEAD
# config.py
import numpy as np

# Simulation Parameters
DT = 0.1           # Time step (seconds)
SIM_TIME = 18.0     # Total simulation time (seconds)

# Road Geometry (Assuming a straight highway for simplicity, curvature kappa = 0)
LANES = [-3.5, 0.0, 3.5]  # y-coordinates of right, center, and left lanes
LANE_WIDTH = 3.5

# Vehicle Constraints
MIN_VELOCITY = 0.0
MAX_VELOCITY = 30.0
MAX_ACCEL = 3.0
MIN_ACCEL = -5.0
MAX_YAW_RATE = 0.5  # rad/s

# Collision Geometry (Effective Ellipse: EV Circle + TV Ellipse)
# Used to check overlap between Ego (s, e_y) and Target (X, Y)
COLLISION_A = 4.0   # Longitudinal semi-axis (meters)
COLLISION_B = 2.0   # Lateral semi-axis (meters)

# ------for making sure that our EV doesn't collide with  the  road ends ---------------------------------------------

# Ego Vehicle physical dimensions
EV_WIDTH = 2.0
EV_LENGTH = 4.5

# Physical road edges
ROAD_EDGE_LEFT = 5.25
ROAD_EDGE_RIGHT = -5.25

# "Safe" boundaries for the EV center point (e_y)
# We add a tiny 0.1m extra buffer so it doesn't scrape the exact edge
MAX_EY = ROAD_EDGE_LEFT - (EV_WIDTH / 2) - 0.1  # 4.15
MIN_EY = ROAD_EDGE_RIGHT + (EV_WIDTH / 2) + 0.1 # -4.15

# ------------------------------------------------------------------------------------------------------------------------
# TV Prediction Modes
# We define distinct modes a TV can execute.
MODE_KEEP_LANE = 0
MODE_CHANGE_LEFT = 1
=======
# config.py
import numpy as np

# Simulation Parameters
DT = 0.1           # Time step (seconds)
SIM_TIME = 18.0     # Total simulation time (seconds)

# Road Geometry (Assuming a straight highway for simplicity, curvature kappa = 0)
LANES = [-3.5, 0.0, 3.5]  # y-coordinates of right, center, and left lanes
LANE_WIDTH = 3.5

# Vehicle Constraints
MIN_VELOCITY = 0.0
MAX_VELOCITY = 30.0
MAX_ACCEL = 3.0
MIN_ACCEL = -5.0
MAX_YAW_RATE = 0.5  # rad/s

# Collision Geometry (Effective Ellipse: EV Circle + TV Ellipse)
# Used to check overlap between Ego (s, e_y) and Target (X, Y)
COLLISION_A = 4.0   # Longitudinal semi-axis (meters)
COLLISION_B = 2.0   # Lateral semi-axis (meters)

# ------for making sure that our EV doesn't collide with  the  road ends ---------------------------------------------

# Ego Vehicle physical dimensions
EV_WIDTH = 2.0
EV_LENGTH = 4.5

# Physical road edges
ROAD_EDGE_LEFT = 5.25
ROAD_EDGE_RIGHT = -5.25

# "Safe" boundaries for the EV center point (e_y)
# We add a tiny 0.1m extra buffer so it doesn't scrape the exact edge
MAX_EY = ROAD_EDGE_LEFT - (EV_WIDTH / 2) - 0.1  # 4.15
MIN_EY = ROAD_EDGE_RIGHT + (EV_WIDTH / 2) + 0.1 # -4.15

# ------------------------------------------------------------------------------------------------------------------------
# TV Prediction Modes
# We define distinct modes a TV can execute.
MODE_KEEP_LANE = 0
MODE_CHANGE_LEFT = 1
>>>>>>> a0716a10d0730f25f94851fde7115e1102c0813c
MODE_CHANGE_RIGHT = 2