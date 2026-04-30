#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from my_py_pkg.robot import Robot
from my_py_pkg.exploration import (
    sense,
    find_frontiers,
    cluster_frontiers,
    communicate,
    compute_Hr,
    compute_Hf,
    is_reachable,
    compute_Htotal,
    assign_goal,
    move
)
from visualization_msgs.msg import Marker, MarkerArray

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        self.marker_pub = self.create_publisher(MarkerArray, 'robot_markers', 10)
        self.robots = [
            Robot(50, 50, 100, rid=0),
            Robot(60, 60, 100, rid=1),
            Robot(40, 70, 100, rid=2),
            Robot(70, 40, 100, rid=3)
        ]
        self.edges = set()
        self.N = 100
        self.global_map = self.create_map()
        self.publisher_ = self.create_publisher(OccupancyGrid, 'map', 10)

        self.timer = self.create_timer(1.0, self.publish_map)

    def create_map(self):
        N = self.N
        global_map = np.ones((N, N))

        global_map[20:80, 20:80] = 0
        global_map[30:60, 5:25] = 0
        global_map[40:75, 80:95] = 0
        global_map[5:25, 40:70] = 0

        global_map[45:55, 25:40] = 0
        global_map[50:60, 70:80] = 0
        global_map[25:40, 55:65] = 0

        global_map[48:52, 25] = 0
        global_map[55, 75:80] = 0
        global_map[25, 60:65] = 0

        global_map[20:25, 30:40] = 1
        global_map[70:75, 50:65] = 1
        global_map[46:50, 80:90] = 1

        global_map[0,:] = 1
        global_map[-1,:] = 1
        global_map[:,0] = 1
        global_map[:,-1] = 1

        return global_map

    def publish_map(self):
        #robot = self.robots[0]
        for r in self.robots:
            sense(r, self.global_map)
        communicate(self.robots, self.edges)
        for robot in self.robots:
            #frontiers = find_frontiers(robot.map)
            #clusters = cluster_frontiers(frontiers)

            goal = assign_goal(robot, self.robots, ds=5)

            if goal is not None:
                self.get_logger().info(f"Robot {robot.id} goal: {goal}")
                move(robot, self.robots, self.global_map)

                #if np.array_equal(robot.pos, robot.last_pos):
                    #robot.stuck_counter += 1
                #else:
                    #robot.stuck_counter = 0

                #robot.last_pos = robot.pos.copy()

        self.get_logger().info(
            f"Robots:{len(self.robots)}, Edges: {len(self.edges)}"
            )
            

        data = self.robots[0].map.flatten()
        msg = OccupancyGrid()

        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = 0.1
        msg.info.width = self.N
        msg.info.height = self.N

        # Convert: 0 → free, 1 → occupied
        data = [
            100 if cell == 1 else 
            0 if cell == 0 else
            -1 
            for cell in data]

        msg.data = data
        self.publish_markers()
        self.publisher_.publish(msg)
        self.get_logger().info("Map published")

    def publish_markers(self):

        marker_array = MarkerArray()

        colors = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 1.0, 0.0),  # green
            (0.0, 0.0, 1.0),  # blue
            (1.0, 1.0, 0.0),  # yellow
        ]

        for i, robot in enumerate(self.robots):

            color = colors[i % len(colors)]

        # ROBOT POSITION
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "robots"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(robot.pos[1]) * 0.1
            marker.pose.position.y = float(robot.pos[0]) * 0.1
            marker.pose.position.z = 0.0

            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        #  GOAL POSITION
            if robot.g_cur is not None:
                goal_marker = Marker()
                goal_marker.header.frame_id = "map"
                goal_marker.header.stamp = self.get_clock().now().to_msg()
                goal_marker.ns = "goals"
                goal_marker.id = i + 100
                goal_marker.type = Marker.CUBE
                goal_marker.action = Marker.ADD

                goal_marker.pose.position.x = float(robot.g_cur[1]) * 0.1
                goal_marker.pose.position.y = float(robot.g_cur[0]) * 0.1
                goal_marker.pose.position.z = 0.0

                goal_marker.scale.x = 0.2
                goal_marker.scale.y = 0.2
                goal_marker.scale.z = 0.2

                goal_marker.color.r = color[0]
                goal_marker.color.g = color[1]
                goal_marker.color.b = color[2]
                goal_marker.color.a = 1.0

                marker_array.markers.append(goal_marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
