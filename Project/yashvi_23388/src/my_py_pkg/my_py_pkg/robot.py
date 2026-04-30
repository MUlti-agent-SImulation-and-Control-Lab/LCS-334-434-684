import numpy as np

class Robot:
    def __init__(self, x, y, map_size, rid=0):
        self.pos = np.array([int(x), int(y)], dtype=int)
        self.prev_pos = self.pos.copy()
        self.stuck_counter = 0
        self.last_pos = self.pos.copy()

        self.map = np.full((map_size, map_size), 0.5)

        self.g_cur = None
        self.g_new = None

        self.steps_since_goal = 0
        self.t_ref = 0

        self.vmax = 1.0

        self.id = rid
