# rl_utils.py
import math
import torch
import numpy as np
import copy
from math import cos, sin, atan2
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan, Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import datetime
import global_vars

# Initialize bridge for converting between OpenCV and ROS images
bridge = CvBridge()

class ModelStateSubscriber(Node):
    """
    Subscribes to TF messages to get the model state (position and orientation).
    """
    def __init__(self):
        super().__init__('model_state_subscriber')
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.listener_callback,
            10)
        
    def listener_callback(self, data):
        pose = data.transforms[1].transform.translation
        orientation = data.transforms[1].transform.rotation

        global_vars.body_pose[0] = pose.x
        global_vars.body_pose[1] = pose.y
        
        # Calculate yaw from quaternion
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w
        global_vars.body_pose[2] = -atan2(2*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2))

class LidarSubscriber(Node):
    """
    Subscribes to laser scan messages to get lidar data.
    """
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/laser_scan',
            self.listener_callback,
            10)
        
        # Store previous lidar data to handle invalid readings
        self.lidar_data_prev_step = np.zeros(20)

    def listener_callback(self, data):
        for i in range(20):
            value = data.ranges[180*i:180*i + 8]
            global_vars.lidar_data[i] = np.max(value)
            
            # If reading is invalid, use previous value
            if global_vars.lidar_data[i] <= 0:
                global_vars.lidar_data[i] = self.lidar_data_prev_step[i]
                
        # Store current readings for next callback
        self.lidar_data_prev_step = copy.copy(global_vars.lidar_data)

class CollisionDetector(Node):
    """
    Detects collisions between the robot and obstacles using image processing.
    """
    def __init__(self, image_size, pixel_to_meter_ratio):
        super().__init__('collision_detector')
        
        # Timer interval for collision detection
        self.time_interval = 0.05

        # Image and robot parameters
        self.image_size = image_size
        self.height = self.image_size
        self.width = self.image_size
        self.center_h = int(self.height/2)
        self.center_w = int(self.width/2)
        self.pixel_to_meter = pixel_to_meter_ratio
        
        # Robot dimensions
        self.robot_length = int(2.8/self.pixel_to_meter)
        self.robot_width = int(1.3/self.pixel_to_meter)
  
        # Add safety margin around robot
        margin = int(0.4/self.pixel_to_meter)
        self.boundary_length = self.robot_length + 2*margin
        self.boundary_width = self.robot_width + 2*margin
        
        # Initialize robot region mask
        self.robot_region = np.zeros((self.height, self.width), np.uint8)

        # Publishers for visualization images
        self.combined_image_pub = self.create_publisher(Image, '/sum_image', 10)
        self.collision_mask_pub = self.create_publisher(Image, '/common_part_image', 10)
        
        # Timer for periodic collision detection
        self.timer = self.create_timer(self.time_interval, self.timer_callback)

    def timer_callback(self):
        # Check if global variables are initialized
        if global_vars.image is None or global_vars.image_for_clash_calc is None:
            self.get_logger().warn("Global image variables not initialized yet")
            return

        # Copy current environment images
        self.current_image = copy.copy(global_vars.image)
        self.collision_image = copy.copy(global_vars.image_for_clash_calc)
        self.robot_region[:,:] = 0

        # Get robot orientation
        theta = global_vars.body_pose[2]
        
        # Calculate outer boundary points (with safety margin)
        self.outer_boundary = self._calculate_boundary_points(theta, self.boundary_length, self.boundary_width)
        pts_outer = np.array(self.outer_boundary, np.int32)
        
        # Draw outer boundary
        cv2.fillPoly(self.current_image, [pts_outer], (80, 80, 80))
        cv2.fillPoly(self.robot_region, [pts_outer], 255)

        # Calculate robot body points
        self.robot_body = self._calculate_boundary_points(theta, self.robot_length, self.robot_width)
        pts_body = np.array(self.robot_body, np.int32)

        # Draw robot body
        cv2.fillPoly(self.current_image, [pts_body], (255, 165, 0))
        
        # Draw front axle line
        front_left = self.robot_body[0]
        front_right = self.robot_body[1]
        cv2.line(self.current_image, front_left, front_right,
                 color=(0, 0, 255), thickness=3, lineType=cv2.LINE_4, shift=0)
        
        # Detect collision
        collision_mask = cv2.bitwise_and(self.robot_region, self.collision_image)
        global_vars.clash_sum = cv2.countNonZero(collision_mask)
 
        # Debug log
        if global_vars.clash_sum > 0:
            self.get_logger().info(f"COLLISION DETECTED! clash_sum: {global_vars.clash_sum}")
        
        # Publish visualization images
        self.collision_mask_pub.publish(bridge.cv2_to_imgmsg(collision_mask))
        self.combined_image_pub.publish(bridge.cv2_to_imgmsg(self.current_image))
        
    def _calculate_boundary_points(self, theta, length, width):
        """
        Calculate the four corner points of a rectangle given its center, dimensions and orientation.
        """
        half_length = length / 2
        half_width = width / 2
        
        # Calculate robot position in image coordinates
        robot_x = self.center_w + global_vars.body_pose[0] / self.pixel_to_meter
        robot_y = self.center_h - global_vars.body_pose[1] / self.pixel_to_meter
        
        # Calculate corner points
        points = [
            # Front-left
            [int(cos(theta)*half_length - sin(theta)*half_width + robot_x),
             int(sin(theta)*half_length + cos(theta)*half_width + robot_y)],
            # Front-right
            [int(cos(theta)*half_length - sin(theta)*(-half_width) + robot_x),
             int(sin(theta)*half_length + cos(theta)*(-half_width) + robot_y)],
            # Rear-right
            [int(cos(theta)*(-half_length) - sin(theta)*(-half_width) + robot_x),
             int(sin(theta)*(-half_length) + cos(theta)*(-half_width) + robot_y)],
            # Rear-left
            [int(cos(theta)*(-half_length) - sin(theta)*half_width + robot_x),
             int(sin(theta)*(-half_length) + cos(theta)*half_width + robot_y)]
        ]
        
        return points

def soft_update(target, source, tau):
    """
    Perform soft update of target network parameters from source network.
    
    Args:
        target: Target network
        source: Source network
        tau: Update rate (0 < tau <= 1)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """
    Copy parameters from source network to target network.
    
    Args:
        target: Target network
        source: Source network
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def save_learning_curve(episodes, rewards, eval_episodes, eval_rewards, filename='learning_curve'):
    """
    Save a plot of the learning curve showing training and evaluation rewards with confidence bands.
    
    Args:
        episodes: List of episode numbers for training
        rewards: List of training episode rewards
        eval_episodes: List of episode numbers for evaluation
        eval_rewards: List of evaluation rewards
        filename: Filename for the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot training rewards
    plt.plot(episodes, rewards, 'b-', alpha=0.2, label='Training Rewards')
    
    # Add a smoothed version of training rewards with confidence bands
    window_size = min(len(rewards) // 5, 10) if len(rewards) > 10 else 1
    if window_size > 1:
        smoothed_rewards = []
        confidence_upper = []
        confidence_lower = []
        
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(rewards), i + window_size + 1)
            window_rewards = rewards[start_idx:end_idx]
            
            # Calculate mean and standard deviation for the window
            mean_reward = sum(window_rewards) / len(window_rewards)
            smoothed_rewards.append(mean_reward)
            
            # Calculate standard deviation for confidence bands
            if len(window_rewards) > 1:
                std_dev = np.std(window_rewards)
                # 95% confidence interval (approximately 1.96 standard deviations)
                confidence_upper.append(mean_reward + 1.96 * std_dev / np.sqrt(len(window_rewards)))
                confidence_lower.append(mean_reward - 1.96 * std_dev / np.sqrt(len(window_rewards)))
            else:
                confidence_upper.append(mean_reward)
                confidence_lower.append(mean_reward)
        
        # Plot smoothed line
        plt.plot(episodes, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed Training (window={window_size*2+1})')
        
        # Plot confidence bands
        plt.fill_between(episodes, confidence_lower, confidence_upper, color='b', alpha=0.2, label='95% Confidence Interval')
    
    # Plot evaluation rewards if available
    if eval_episodes and eval_rewards:
        plt.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluation Rewards')
        
        # If we have multiple evaluation points, add a smoothed trend line
        if len(eval_rewards) > 3:
            eval_smooth = []
            for i in range(len(eval_rewards)):
                start_idx = max(0, i - 1)
                end_idx = min(len(eval_rewards), i + 2)
                eval_smooth.append(sum(eval_rewards[start_idx:end_idx]) / (end_idx - start_idx))
            plt.plot(eval_episodes, eval_smooth, 'r--', linewidth=1.5, alpha=0.7, label='Evaluation Trend')
    
    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('SAC Learning Curve - Obstacle Avoidance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    curves_dir = 'learning_curves'
    if not os.path.exists(curves_dir):
        os.makedirs(curves_dir)
        
    # Add timestamp to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = f"{curves_dir}/{filename}_{timestamp}.png"
    
    plt.savefig(full_path)
    print(f"Learning curve saved to {full_path}")
    
    # Also save data as CSV for later analysis
    csv_path = f"{curves_dir}/{filename}_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("episode,training_reward,smoothed_reward,confidence_lower,confidence_upper,eval_episode,eval_reward\n")
        
        # Save all data including smoothed values and confidence bands
        for i in range(len(episodes)):
            ep = episodes[i]
            rw = rewards[i]
            smooth = smoothed_rewards[i] if i < len(smoothed_rewards) else ""
            c_lower = confidence_lower[i] if i < len(confidence_lower) else ""
            c_upper = confidence_upper[i] if i < len(confidence_upper) else ""
            
            # Find matching evaluation episode, if any
            eval_ep = ""
            eval_rw = ""
            if eval_episodes:
                # Find closest evaluation episode
                if episodes[i] in eval_episodes:
                    idx = eval_episodes.index(episodes[i])
                    eval_ep = eval_episodes[idx]
                    eval_rw = eval_rewards[idx]
            
            f.write(f"{ep},{rw},{smooth},{c_lower},{c_upper},{eval_ep},{eval_rw}\n")
            
    print(f"Learning curve data saved to {csv_path}")