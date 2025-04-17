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
import weakref

# Initialize bridge for converting between OpenCV and ROS images
bridge = CvBridge()

class ModelStateSubscriber(Node):
    """
    Subscribes to TF messages to get the model state (position and orientation).
    Memory-optimized version.
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

        # Update global state directly without temporary variables
        global_vars.body_pose[0] = pose.x
        global_vars.body_pose[1] = pose.y
        
        # Calculate yaw from quaternion - more efficient calculation
        global_vars.body_pose[2] = -atan2(
            2*(orientation.x*orientation.y + orientation.z*orientation.w), 
            (orientation.x**2 - orientation.y**2 - orientation.z**2 + orientation.w**2)
        )

class LidarSubscriber(Node):
    """
    Subscribes to laser scan messages to get lidar data.
    Memory-optimized version.
    """
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/laser_scan',
            self.listener_callback,
            10)
        
        # Store previous lidar data to handle invalid readings
        self.lidar_data_prev_step = np.zeros(20, dtype=np.float32)

    def listener_callback(self, data):
        # Process data in chunks for better memory efficiency
        for i in range(20):
            # Get slice of range data
            start_idx = 180*i
            end_idx = start_idx + 8
            value = data.ranges[start_idx:end_idx]
            
            # Find maximum value (avoid creating unnecessary lists/arrays)
            max_val = 0.0
            for j in range(len(value)):
                if value[j] > max_val:
                    max_val = value[j]
            
            # Update global variable
            global_vars.lidar_data[i] = max_val if max_val > 0 else self.lidar_data_prev_step[i]
                
        # Copy current readings for next callback - use efficient numpy operations
        np.copyto(self.lidar_data_prev_step, global_vars.lidar_data)

class CollisionDetector(Node):
    """
    Detects collisions between the robot and obstacles using image processing.
    Memory-optimized version.
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
        
        # Initialize robot region mask - use memory-mapped file
        self._create_robot_region_mask()

        # Publishers for visualization images
        self.combined_image_pub = self.create_publisher(Image, '/sum_image', 10)
        self.collision_mask_pub = self.create_publisher(Image, '/common_part_image', 10)
        
        # Cache for sin/cos calculations
        self.sin_cos_cache = {}
        
        # Re-use arrays for collision detection
        self.current_image = None
        self.collision_image = None
        
        # Timer for periodic collision detection
        self.timer = self.create_timer(self.time_interval, self.timer_callback)
    
    def _create_robot_region_mask(self):
        """Create memory-mapped array for robot region mask"""
        os.makedirs('temp', exist_ok=True)
        robot_region_file = 'temp/robot_region.dat'
        
        try:
            # Create memory-mapped array
            self.robot_region = np.memmap(robot_region_file, dtype=np.uint8, mode='w+', 
                                        shape=(self.height, self.width))
            self.robot_region.fill(0)
            self.robot_region_file = robot_region_file
        except Exception as e:
            self.get_logger().error(f"Error creating memory-mapped array for robot region: {e}")
            # Fall back to regular NumPy array
            self.robot_region = np.zeros((self.height, self.width), np.uint8)

    def timer_callback(self):
        # Check if global variables are initialized
        if global_vars.image is None or global_vars.image_for_clash_calc is None:
            self.get_logger().warn("Global image variables not initialized yet")
            return

        # Reuse arrays if already created, otherwise create new ones
        if self.current_image is None:
            self.current_image = np.copy(global_vars.image)
        else:
            np.copyto(self.current_image, global_vars.image)
            
        if self.collision_image is None:
            self.collision_image = np.copy(global_vars.image_for_clash_calc)
        else:
            np.copyto(self.collision_image, global_vars.image_for_clash_calc)
            
        # Clear robot region efficiently
        self.robot_region.fill(0)

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
        
        # Detect collision efficiently
        collision_mask = cv2.bitwise_and(self.robot_region, self.collision_image)
        global_vars.clash_sum = cv2.countNonZero(collision_mask)
 
        # Debug log
        if global_vars.clash_sum > 0:
            self.get_logger().info(f"COLLISION DETECTED! clash_sum: {global_vars.clash_sum}")
        
        # Publish visualization images - only copy when needed
        self.collision_mask_pub.publish(bridge.cv2_to_imgmsg(collision_mask))
        self.combined_image_pub.publish(bridge.cv2_to_imgmsg(self.current_image))
        
        # Flush memory-mapped data if applicable
        if hasattr(self.robot_region, 'flush'):
            self.robot_region.flush()
        
    def _calculate_boundary_points(self, theta, length, width):
        """
        Calculate the four corner points with caching for repeated calculations.
        """
        # Check cache for sin/cos values (rounded to 3 decimal places for cache efficiency)
        theta_key = round(theta, 3)
        if theta_key not in self.sin_cos_cache:
            self.sin_cos_cache[theta_key] = (sin(theta_key), cos(theta_key))
            
            # Limit cache size to prevent memory growth
            if len(self.sin_cos_cache) > 1000:
                # Remove oldest entries (this is efficient enough for this use case)
                keys_to_remove = list(self.sin_cos_cache.keys())[:-500]
                for key in keys_to_remove:
                    del self.sin_cos_cache[key]
                
        # Get cached values
        sin_theta, cos_theta = self.sin_cos_cache[theta_key]
        
        half_length = length / 2
        half_width = width / 2
        
        # Calculate robot position in image coordinates
        robot_x = self.center_w + global_vars.body_pose[0] / self.pixel_to_meter
        robot_y = self.center_h - global_vars.body_pose[1] / self.pixel_to_meter
        
        # Pre-calculate repeated terms
        cos_hl = cos_theta * half_length
        sin_hl = sin_theta * half_length
        cos_hw = cos_theta * half_width
        sin_hw = sin_theta * half_width
        
        # Calculate corner points without duplicating calculations
        points = [
            # Front-left
            [int(cos_hl - sin_hw + robot_x), int(sin_hl + cos_hw + robot_y)],
            # Front-right
            [int(cos_hl + sin_hw + robot_x), int(sin_hl - cos_hw + robot_y)],
            # Rear-right
            [int(-cos_hl + sin_hw + robot_x), int(-sin_hl - cos_hw + robot_y)],
            # Rear-left
            [int(-cos_hl - sin_hw + robot_x), int(-sin_hl + cos_hw + robot_y)]
        ]
        
        return points
        
    def __del__(self):
        """Clean up resources when object is deleted"""
        try:
            # Clean up memory-mapped file
            if hasattr(self, 'robot_region_file') and os.path.exists(self.robot_region_file):
                os.remove(self.robot_region_file)
        except Exception as e:
            print(f"Error cleaning up collision detector resources: {e}")

def soft_update(target, source, tau):
    """
    Perform soft update of target network parameters from source network.
    Memory-optimized implementation.
    
    Args:
        target: Target network
        source: Source network
        tau: Update rate (0 < tau <= 1)
    """
    with torch.no_grad():  # Avoid storing gradients to save memory
        for target_param, param in zip(target.parameters(), source.parameters()):
            # Inplace operations to save memory
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(param.data, alpha=tau)

def hard_update(target, source):
    """
    Copy parameters from source network to target network.
    Memory-optimized implementation.
    
    Args:
        target: Target network
        source: Source network
    """
    with torch.no_grad():  # Avoid storing gradients
        for target_param, param in zip(target.parameters(), source.parameters()):
            # Direct copy without creating intermediate tensors
            target_param.data.copy_(param.data)

def evaluate_policy(agent, env, max_steps, episode_num=0):
    """
    Evaluate the agent's policy without storing gradients to save memory.
    
    Args:
        agent: agent
        env: Environment
        max_steps: Maximum episode steps
        episode_num: Current evaluation episode number
        
    Returns:
        float: Episode reward
    """
    # Use no_grad mode to save memory
    with torch.no_grad():
        state = env.reset()
        episode_reward = 0
        eval_steps = 0
        done = False
        
        while not done:
            # Select action deterministically
            action = agent.select_action(state, evaluate=True)
            
            # Take step in environment
            next_state, reward, done = env.step(action, eval_steps, max_steps)
            
            episode_reward += reward
            eval_steps += 1
            state = next_state
            
            # Print progress for long evaluations
            if eval_steps % 20 == 0:
                print(f"Evaluation episode {episode_num}, step {eval_steps}, reward so far: {episode_reward}")
    
    return episode_reward

def save_learning_curve(episodes, rewards, eval_episodes, eval_rewards, filename='learning_curve'):
    """
    Save a plot of the learning curve showing training and evaluation rewards with confidence bands.
    Memory-optimized implementation.
    
    Args:
        episodes: List of episode numbers for training
        rewards: List of training episode rewards
        eval_episodes: List of episode numbers for evaluation
        eval_rewards: List of evaluation rewards
        filename: Filename for the saved plot
    """
    # Convert lists to numpy arrays for more efficient operations
    episodes = np.array(episodes, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    
    # Create figure with reduced DPI to save memory
    plt.figure(figsize=(10, 6), dpi=80)
    
    # Plot training rewards
    plt.plot(episodes, rewards, 'b-', alpha=0.2, label='Training Rewards')
    
    # Add a smoothed version of training rewards with confidence bands
    window_size = min(len(rewards) // 5, 10) if len(rewards) > 10 else 1
    if window_size > 1:
        # Pre-allocate arrays for smoothed values and confidence bands
        smoothed_rewards = np.zeros_like(rewards)
        confidence_upper = np.zeros_like(rewards)
        confidence_lower = np.zeros_like(rewards)
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000  # Process 1000 points at a time
        for chunk_start in range(0, len(rewards), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(rewards))
            chunk_indices = np.arange(chunk_start, chunk_end)
            
            for i in chunk_indices:
                start_idx = max(0, i - window_size)
                end_idx = min(len(rewards), i + window_size + 1)
                window_rewards = rewards[start_idx:end_idx]
                
                # Calculate mean and standard deviation for the window
                mean_reward = np.mean(window_rewards)
                smoothed_rewards[i] = mean_reward
                
                # Calculate standard deviation for confidence bands
                if len(window_rewards) > 1:
                    std_dev = np.std(window_rewards)
                    # 95% confidence interval
                    confidence_upper[i] = mean_reward + 1.96 * std_dev / np.sqrt(len(window_rewards))
                    confidence_lower[i] = mean_reward - 1.96 * std_dev / np.sqrt(len(window_rewards))
                else:
                    confidence_upper[i] = mean_reward
                    confidence_lower[i] = mean_reward
        
        # Plot smoothed line
        plt.plot(episodes, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed Training (window={window_size*2+1})')
        
        # Plot confidence bands
        plt.fill_between(episodes, confidence_lower, confidence_upper, color='b', alpha=0.2, label='95% Confidence Interval')
    
    # Plot evaluation rewards if available
    if eval_episodes and eval_rewards:
        eval_episodes = np.array(eval_episodes, dtype=np.float32)
        eval_rewards = np.array(eval_rewards, dtype=np.float32)
        
        plt.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluation Rewards')
        
        # If we have multiple evaluation points, add a smoothed trend line
        if len(eval_rewards) > 3:
            eval_smooth = np.zeros_like(eval_rewards)
            for i in range(len(eval_rewards)):
                start_idx = max(0, i - 1)
                end_idx = min(len(eval_rewards), i + 2)
                eval_smooth[i] = np.mean(eval_rewards[start_idx:end_idx])
            plt.plot(eval_episodes, eval_smooth, 'r--', linewidth=1.5, alpha=0.7, label='Evaluation Trend')
    
    # Add labels and title with optimized settings
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('SAC Learning Curve - Obstacle Avoidance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use a tight layout to optimize figure size
    plt.tight_layout()
    
    # Save the figure with reduced quality to save disk space
    curves_dir = 'learning_curves'
    if not os.path.exists(curves_dir):
        os.makedirs(curves_dir)
        
    # Add timestamp to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = f"{curves_dir}/{filename}_{timestamp}.png"
    
    plt.savefig(full_path, dpi=80, bbox_inches='tight', pad_inches=0.1, optimize=True)
    print(f"Learning curve saved to {full_path}")
    
    # Release memory
    plt.close()
    
    # Also save data as CSV for later analysis - use efficient writing
    csv_path = f"{curves_dir}/{filename}_{timestamp}.csv"
    with open(csv_path, 'w') as f:
        f.write("episode,training_reward,smoothed_reward,confidence_lower,confidence_upper,eval_episode,eval_reward\n")
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000
        for chunk_start in range(0, len(episodes), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(episodes))
            
            # Prepare chunk data
            csv_lines = []
            for i in range(chunk_start, chunk_end):
                ep = episodes[i]
                rw = rewards[i]
                smooth = smoothed_rewards[i] if 'smoothed_rewards' in locals() else ""
                c_lower = confidence_lower[i] if 'confidence_lower' in locals() else ""
                c_upper = confidence_upper[i] if 'confidence_upper' in locals() else ""
                
                # Find matching evaluation episode, if any
                eval_ep = ""
                eval_rw = ""
                if len(eval_episodes) > 0:
                    # Find closest evaluation episode using efficient numpy operations
                    idx = np.abs(eval_episodes - ep).argmin()
                    if abs(eval_episodes[idx] - ep) < 1e-6:  # Close enough to be considered equal
                        eval_ep = eval_episodes[idx]
                        eval_rw = eval_rewards[idx]
                
                csv_lines.append(f"{ep},{rw},{smooth},{c_lower},{c_upper},{eval_ep},{eval_rw}\n")
            
            # Write chunk in one operation
            f.writelines(csv_lines)
            
    print(f"Learning curve data saved to {csv_path}")