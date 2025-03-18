import math
import torch

import numpy as np
import copy
import math
import cv2
from math import cos, sin, atan2
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import LaserScan, Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge

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
        global body_pose

        pose = data.transforms[1].transform.translation
        orientation = data.transforms[1].transform.rotation

        body_pose[0] = pose.x
        body_pose[1] = pose.y
        
        # Calculate yaw from quaternion
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w
        body_pose[2] = -atan2(2*(q0*q1 + q2*q3), (q0**2 - q1**2 - q2**2 + q3**2))

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
        self.prev_lidar_data = np.zeros(20)

    def listener_callback(self, data):
        global lidar_data

        for i in range(20):
            value = data.ranges[180*i:180*i + 8]
            lidar_data[i] = np.max(value)
            
            # If reading is invalid, use previous value
            if lidar_data[i] <= 0:
                lidar_data[i] = self.prev_lidar_data[i]
                
        # Store current readings for next callback
        self.prev_lidar_data = copy.copy(lidar_data)

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
        global body_pose, clash_sum

        # Copy current environment images
        self.current_image = copy.copy(image)
        self.collision_image = copy.copy(image_for_clash_calc)
        self.robot_region[:,:] = 0

        # Get robot orientation
        theta = body_pose[2]
        
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
        clash_sum = cv2.countNonZero(collision_mask)
 
        # Publish visualization images
        self.collision_mask_pub.publish(bridge.cv2_to_imgmsg(collision_mask))
        self.combined_image_pub.publish(bridge.cv2_to_imgmsg(self.current_image))
        
    def _calculate_boundary_points(self, theta, length, width):
        """
        Calculate the four corner points of a rectangle given its center, dimensions and orientation.
        
        Args:
            theta: Orientation angle
            length: Length of rectangle
            width: Width of rectangle
            
        Returns:
            List of four corner points
        """
        half_length = length / 2
        half_width = width / 2
        
        # Calculate robot position in image coordinates
        robot_x = self.center_w + body_pose[0] / self.pixel_to_meter
        robot_y = self.center_h - body_pose[1] / self.pixel_to_meter
        
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


# RL Algorithm utility functions

def create_log_gaussian(mean, log_std, t):
    """
    Create log of Gaussian distribution.
    
    Args:
        mean: Mean of the distribution
        log_std: Log of standard deviation
        t: Input tensor
        
    Returns:
        Log probability
    """
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    """
    Compute log of sum of exponentials in a numerically stable way.
    
    Args:
        inputs: Input tensor
        dim: Dimension to reduce
        keepdim: Whether to keep the reduced dimension
        
    Returns:
        Result of log-sum-exp operation
    """
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

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