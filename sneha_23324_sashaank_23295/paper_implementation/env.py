# env.py
import numpy as np
import config
from dynamics import EgoVehicle, TargetVehicle

class HighwayEnv:
    def __init__(self):
        self.ego = None
        self.targets = []
        self.time = 0.0

    def reset(self):
        self.time = 0.0
        
        # EV starts at s=0, center lane (e_y=0), looking straight (e_psi=0), 20m/s
        self.ego = EgoVehicle(s=0.0, e_y=config.LANES[1], e_psi=0.0, v=20.0)
        
        # Target 1 ahead in center lane, Target 2 in right lane
        self.targets = [
            TargetVehicle(x=30.0, y=config.LANES[1], vx=15.0, tv_id="TV_1"),
            TargetVehicle(x=15.0, y=config.LANES[0], vx=18.0, tv_id="TV_2")
        ]
        return self.get_state_dict()

    def step(self, ego_action, tv_modes):
        """
        ego_action: [a, psi_dot]
        tv_modes: list of modes [mode_tv1, mode_tv2]
        """
        # Step EV
        self.ego.step(ego_action)

        # Step TVs
        for i, target in enumerate(self.targets):
            target.step(tv_modes[i])

        self.time += config.DT
        collision = self._check_collision()
        done = collision or (self.time >= config.SIM_TIME)

        return self.get_state_dict(), done, {"collision": collision}

    def _check_collision(self):
        # EV Frenet (s, e_y) maps to Global (X, Y) on a straight road
        ego_x, ego_y = self.ego.state[0], self.ego.state[1]
        
        for target in self.targets:
            tv_x, tv_y = target.state[0], target.state[1]
            dx = ego_x - tv_x
            dy = ego_y - tv_y
            
            # Elliptical collision boundary (Equation representation)
            if (dx**2 / config.COLLISION_A**2) + (dy**2 / config.COLLISION_B**2) <= 1.0:
                return True
        return False

    def get_state_dict(self):
        return {
            "ego": self.ego.get_state(),
            "targets": [t.get_state() for t in self.targets]
        }
    