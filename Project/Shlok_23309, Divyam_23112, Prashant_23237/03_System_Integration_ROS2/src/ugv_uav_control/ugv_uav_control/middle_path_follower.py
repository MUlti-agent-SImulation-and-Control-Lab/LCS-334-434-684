"""
Phase 3: Mission Integration & Validation (ROS 2)
Component: Perception & Vision Pipeline
Purpose: Extracts the UGV's relative path from the downward-facing camera.
         Performs ArUco marker detection (ID 55) for EKF initialization 
         and identifies the lane centerline for waypoint generation.
Inputs: /overhead_camera/image (sensor_msgs/Image)
Outputs: /waypoints (PoseStamped), /aruco_pose (PoseStamped)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    q = [0] * 4
    q[0] = cy * cp * sr - sy * sp * cr
    q[1] = sy * cp * sr + cy * sp * cr
    q[2] = sy * cp * cr - cy * sp * sr
    q[3] = cy * cp * cr + sy * sp * sr
    return q
class MiddlePathFollower(Node):
    def __init__(self):
        super().__init__('middle_path_follower')
        self.get_logger().info('Initializing MiddlePathFollower...')
        
        self.subscription = self.create_subscription(Image, 'overhead_camera/image', self.image_callback, 10)
        self.debug_pub = self.create_publisher(Image, '/middle_path/debug_image', 10)
        self.waypoint_pub = self.create_publisher(PoseStamped, '/waypoints', 10) 
        self.nearest_pub = self.create_publisher(PoseStamped, '/nearest_point', 10)
        self.aruco_pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters()
            
        self.aruco_params.minMarkerPerimeterRate = 0.01
        
        self.camera_height = 5.0
        self.fov = 1.047 # 60 degrees
        self.image_width = None
        self.scale = None
        self.frame_count = 0
        self.get_logger().info('Middle Path Follower Node Started')
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'cv_bridge exception: {e}')
            return
        if self.image_width is None:
            self.image_height, self.image_width = cv_image.shape[:2]
            world_width = 2 * self.camera_height * math.tan(self.fov / 2)
            self.scale = world_width / self.image_width
            self.get_logger().info(f'Camera Calibrated. Scale: {self.scale:.5f} m/px')
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        
        robot_pos = None
        robot_angle = 0.0
        
        if ids is not None:
            self.get_logger().info(f'DETECTED IDS: {ids}') 

            for i in range(len(ids)):
                if ids[i][0] == 55: # Robot Marker ID
                    c = corners[i][0]
                    cx = int(np.mean(c[:, 0]))
                    cy = int(np.mean(c[:, 1]))
                    robot_pos = (cx, cy)
                    top_mid_x = (c[0][0] + c[1][0]) / 2
                    top_mid_y = (c[0][1] + c[1][1]) / 2
                    robot_angle = math.atan2(top_mid_y - cy, top_mid_x - cx)
                    break
                
        # Publish ArUco pose for EKF
        if robot_pos is not None:
            aruco_msg = PoseStamped()
            aruco_msg.header.stamp = self.get_clock().now().to_msg()
            aruco_msg.header.frame_id = 'world'
            # Convert pixel to world coordinates
            aruco_msg.pose.position.x = (robot_pos[0] - self.image_width/2) * self.scale
            aruco_msg.pose.position.y = (robot_pos[1] - self.image_height/2) * self.scale
            q = quaternion_from_euler(0, 0, robot_angle)
            aruco_msg.pose.orientation.x, aruco_msg.pose.orientation.y = q[0], q[1]
            aruco_msg.pose.orientation.z, aruco_msg.pose.orientation.w = q[2], q[3]
            self.aruco_pose_pub.publish(aruco_msg)
        
        if robot_pos is None:
            return
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        
        h, w = thresh.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        
        # Seed search ahead of robot
        seed_point = robot_pos
        found_seed = False
        search_dists = [40, 50, 60, 30, 70] 
        search_angles = [0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6]
        
        for dist in search_dists:
            for angle_offset in search_angles:
                check_angle = robot_angle + angle_offset
                sx = int(robot_pos[0] + dist * math.cos(check_angle))
                sy = int(robot_pos[1] + dist * math.sin(check_angle))
                sx = max(0, min(w-1, sx))
                sy = max(0, min(h-1, sy))
                if thresh[sy, sx] == 255:
                    seed_point = (sx, sy)
                    found_seed = True
                    break
            if found_seed: break
            
        lane_mask_img = thresh.copy()
        cv2.floodFill(lane_mask_img, flood_mask, seed_point, 128)
        lane_mask = np.zeros_like(thresh)
        lane_mask[lane_mask_img == 128] = 255
        
        dist_map = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        dist_map[lane_mask == 0] = 0
        
        # Visualization
        min_val, max_val, _, _ = cv2.minMaxLoc(dist_map)
        dist_norm = dist_map / max_val if max_val > 0 else dist_map
        heatmap = cv2.applyColorMap((dist_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        final_img = cv2.addWeighted(cv_image, 0.6, heatmap, 0.4, 0)
        
        # Calculate Target
        lookahead_dist = 50
        proj_x = robot_pos[0] + lookahead_dist * math.cos(robot_angle)
        proj_y = robot_pos[1] + lookahead_dist * math.sin(robot_angle)
        
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(proj_x), int(proj_y)), 60, 255, -1)
        
        masked_dist = dist_map.copy()
        masked_dist[mask == 0] = 0
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_dist)
        
        target_point = max_loc if max_val > 0 else robot_pos
        # Debug Draw
        cv2.circle(final_img, robot_pos, 8, (0, 0, 255), -1) 
        cv2.circle(final_img, target_point, 8, (0, 255, 255), -1) 
        cv2.line(final_img, robot_pos, target_point, (255, 0, 255), 2)
        
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            h, w = cv_image.shape[:2]
            half_h, half_w = h // 3, w // 3
            # 2x2 Grid
            thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            mask_bgr = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
            top_row = np.hstack((cv2.resize(thresh_bgr, (half_w, half_h)), cv2.resize(mask_bgr, (half_w, half_h))))
            bot_row = np.hstack((cv2.resize(heatmap, (half_w, half_h)), cv2.resize(final_img, (half_w, half_h))))
            debug_img = np.vstack((top_row, bot_row))
            try:
                cv2.imshow("Middle Path Follower", debug_img)
                cv2.waitKey(1)
            except: pass
        # Publish Waypoint
        if self.scale is not None:
             v_x = target_point[0] - robot_pos[0]
             v_y = target_point[1] - robot_pos[1]
             dist_meters = math.sqrt(v_x**2 + v_y**2) * self.scale
             
             target_angle_image = math.atan2(v_y, v_x)
             relative_angle = -(target_angle_image - robot_angle)
             relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
             
             waypoint_x = dist_meters * math.cos(relative_angle)
             waypoint_y = dist_meters * math.sin(relative_angle)
             
             q = quaternion_from_euler(0, 0, relative_angle)
             pose_msg = PoseStamped()
             pose_msg.header.stamp = self.get_clock().now().to_msg()
             pose_msg.header.frame_id = 'base_link'
             pose_msg.pose.position.x = waypoint_x
             pose_msg.pose.position.y = waypoint_y
             pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = q
             self.get_logger().info(f'Publishing Waypoint: dist={dist_meters:.3f}m, angle={relative_angle:.3f}rad')
             self.waypoint_pub.publish(pose_msg)
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MiddlePathFollower())
    rclpy.shutdown()
if __name__ == '__main__':
    main()
