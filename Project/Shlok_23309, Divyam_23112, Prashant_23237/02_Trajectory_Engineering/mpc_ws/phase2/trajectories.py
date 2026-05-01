"""
Phase 2 trajectory definitions.

Loaded by waypoint_publisher_node via importlib (the publisher takes a
trajectories_path parameter pointing at this file).

Each trajectory is a list of [x, y, z] waypoints in metres in the world frame.
Constant-altitude trajectories at z = 1.0 m to match the Phase 2 hover-hold
takeoff height (smooth handoff: hover at z=1.0, first waypoint at z=1.0).
"""

M_SHAPE = [
    [0.0, 0.0, 1.0],   # bottom-left start
    [0.0, 2.0, 1.0],   # top-left
    [1.0, 1.0, 1.0],   # middle dip
    [2.0, 2.0, 1.0],   # top-right
    [2.0, 0.0, 1.0],   # bottom-right end
]

S_SHAPE = [
    [0.0,  0.0, 1.0],   # bottom
    [1.0,  0.5, 1.0],   # curve right
    [0.0,  1.0, 1.0],   # middle cross
    [-1.0, 1.5, 1.0],   # curve left
    [0.0,  2.0, 1.0],   # top
]

TRAJECTORIES = {
    "M": M_SHAPE,
    "S": S_SHAPE,
}
