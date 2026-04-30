# Functionality: Encapsulates the track math. It imports the parameters from config.py 
# and provides the function needed by the animation loop to draw the road and vehicles.
import numpy as np
import config

def get_cartesian(s, ey):
    """
    Converts Frenet (s, ey) to Cartesian (X, Y, yaw) for the oval track.
    """
    s = s % config.TRACK_LENGTH 
    
    if s < config.S1:
        x, y, yaw = s, 0, 0.0
    elif s < config.S2:
        angle = (s - config.S1) / config.CURVE_RADIUS
        x = config.S1 + config.CURVE_RADIUS * np.sin(angle)
        y = config.CURVE_RADIUS * (1 - np.cos(angle))
        yaw = angle
    elif s < config.S3:
        x = config.S1 - (s - config.S2)
        y = 2 * config.CURVE_RADIUS
        yaw = np.pi
    else:
        angle = np.pi + (s - config.S3) / config.CURVE_RADIUS
        x = config.CURVE_RADIUS * np.sin(angle)
        y = config.CURVE_RADIUS * (1 - np.cos(angle))
        yaw = angle
        
    X = x - ey * np.sin(yaw)
    Y = y + ey * np.cos(yaw)
    
    return X, Y, yaw