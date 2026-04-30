<<<<<<< HEAD
# env.py
import numpy as np
import cartersian_coordinates.config as config
from cartersian_coordinates.dynamics import Vehicle

class HighwayEnv:
    def __init__(self):
        self.ego = None
        self.targets = []
        self.time = 0.0
        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.time = 0.0
        
        # Initialize ego vehicle in the middle lane
        self.ego = Vehicle(x=0.0, y=config.LANES[1], v=20.0, vehicle_id="ego")
        
        # Initialize 2 target vehicles ahead of ego
        self.targets = [
            Vehicle(x=30.0, y=config.LANES[1], v=15.0, vehicle_id="target_1"), # Slower vehicle ahead
            Vehicle(x=15.0, y=config.LANES[2], v=18.0, vehicle_id="target_2")  # Vehicle in right lane
        ]
        
        return self.get_full_state()

    def step(self, ego_action, target_actions):
        """
        ego_action: [acceleration, lane_change_rate]
        target_actions: list of [acceleration, lane_change_rate] for each target
        """
        # 1. Update Ego Vehicle
        self.ego.step(ego_action[0], ego_action[1])

        # 2. Update Target Vehicles
        for i, target in enumerate(self.targets):
            target.step(target_actions[i][0], target_actions[i][1])

        self.time += config.DT

        # 3. Check for collisions
        collision = self._check_collision()

        # 4. Calculate reward
        reward = self._calculate_reward(collision)

        # 5. Check if episode is done
        done = collision or (self.time >= config.SIM_TIME)

        return self.get_full_state(), reward, done, {"collision": collision}

    # def _check_collision(self):
    #     """Checks Euclidean distance between ego and all targets."""
    #     for target in self.targets:
    #         dist = np.hypot(self.ego.x - target.x, self.ego.y - target.y)
    #         if dist <= config.COLLISION_RADIUS:
    #             return True
    #     return False
    def _check_collision(self):
        """Checks if ego point is inside the effective target ellipse."""
        for target in self.targets:
            dx = self.ego.x - target.x
            dy = self.ego.y - target.y
            
            # Ellipse equation
            if (dx**2 / config.ELLIPSE_A**2) + (dy**2 / config.ELLIPSE_B**2) <= 1.0:
                return True
        return False

    def _calculate_reward(self, collision):
        """Calculates the reward based on progress, lane keeping, and safety."""
        if collision:
            return config.W_COLLISION

        # Forward progress reward (encourages maintaining speed)
        progress_reward = config.W_PROGRESS * self.ego.v

        # Lane keeping penalty (find distance to the nearest lane center)
        distances_to_lanes = [abs(self.ego.y - lane_y) for lane_y in config.LANES]
        min_lane_dist = min(distances_to_lanes)
        lane_penalty = config.W_LANE * (min_lane_dist ** 2)

        return progress_reward + lane_penalty

    def get_full_state(self):
        """Returns a dictionary containing the states of all vehicles."""
        state = {
            "ego": self.ego.get_state(),
            "targets": [t.get_state() for t in self.targets]
        }
=======
# env.py
import numpy as np
import cartersian_coordinates.config as config
from cartersian_coordinates.dynamics import Vehicle

class HighwayEnv:
    def __init__(self):
        self.ego = None
        self.targets = []
        self.time = 0.0
        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.time = 0.0
        
        # Initialize ego vehicle in the middle lane
        self.ego = Vehicle(x=0.0, y=config.LANES[1], v=20.0, vehicle_id="ego")
        
        # Initialize 2 target vehicles ahead of ego
        self.targets = [
            Vehicle(x=30.0, y=config.LANES[1], v=15.0, vehicle_id="target_1"), # Slower vehicle ahead
            Vehicle(x=15.0, y=config.LANES[2], v=18.0, vehicle_id="target_2")  # Vehicle in right lane
        ]
        
        return self.get_full_state()

    def step(self, ego_action, target_actions):
        """
        ego_action: [acceleration, lane_change_rate]
        target_actions: list of [acceleration, lane_change_rate] for each target
        """
        # 1. Update Ego Vehicle
        self.ego.step(ego_action[0], ego_action[1])

        # 2. Update Target Vehicles
        for i, target in enumerate(self.targets):
            target.step(target_actions[i][0], target_actions[i][1])

        self.time += config.DT

        # 3. Check for collisions
        collision = self._check_collision()

        # 4. Calculate reward
        reward = self._calculate_reward(collision)

        # 5. Check if episode is done
        done = collision or (self.time >= config.SIM_TIME)

        return self.get_full_state(), reward, done, {"collision": collision}

    # def _check_collision(self):
    #     """Checks Euclidean distance between ego and all targets."""
    #     for target in self.targets:
    #         dist = np.hypot(self.ego.x - target.x, self.ego.y - target.y)
    #         if dist <= config.COLLISION_RADIUS:
    #             return True
    #     return False
    def _check_collision(self):
        """Checks if ego point is inside the effective target ellipse."""
        for target in self.targets:
            dx = self.ego.x - target.x
            dy = self.ego.y - target.y
            
            # Ellipse equation
            if (dx**2 / config.ELLIPSE_A**2) + (dy**2 / config.ELLIPSE_B**2) <= 1.0:
                return True
        return False

    def _calculate_reward(self, collision):
        """Calculates the reward based on progress, lane keeping, and safety."""
        if collision:
            return config.W_COLLISION

        # Forward progress reward (encourages maintaining speed)
        progress_reward = config.W_PROGRESS * self.ego.v

        # Lane keeping penalty (find distance to the nearest lane center)
        distances_to_lanes = [abs(self.ego.y - lane_y) for lane_y in config.LANES]
        min_lane_dist = min(distances_to_lanes)
        lane_penalty = config.W_LANE * (min_lane_dist ** 2)

        return progress_reward + lane_penalty

    def get_full_state(self):
        """Returns a dictionary containing the states of all vehicles."""
        state = {
            "ego": self.ego.get_state(),
            "targets": [t.get_state() for t in self.targets]
        }
>>>>>>> a0716a10d0730f25f94851fde7115e1102c0813c
        return state