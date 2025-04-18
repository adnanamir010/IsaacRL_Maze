import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import math
import os


class VectorizedDDEnv(gym.Env):
    """
    A vectorizable environment for Differential Drive robot navigation.
    This environment simulates a robot navigating through obstacles to reach a goal.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, stage_size=720, obstacle_size=8.0, obstacle_shape="circular"):
        """
        Initialize the robot environment.
        
        Args:
            render_mode: Mode for rendering the environment ('human', 'rgb_array', or None)
            stage_size: Size of the stage in pixels (square)
            obstacle_size: Size of the obstacle cubes
            obstacle_shape: Shape of obstacles ('circular' or 'square')
        """
        super().__init__()
        
        # Environment parameters
        self.stage_size = stage_size
        self.pixel_to_meter = 7.5        # Conversion from pixels to meters
        self.stage_width = 96.0          # Stage width in meters (matches original)
        self.stage_height = 96.0         # Stage height in meters (matches original)
        self.obstacle_size = obstacle_size
        self.obstacle_shape = obstacle_shape  # Shape of obstacles ('circular' or 'square')
        
        # Robot physical parameters
        self.track_width = 1.1           # Track width between wheels
        self.wheelbase = 1.8             # Wheelbase length
        self.wheel_radius = 0.3          # Wheel radius
        self.linear_velocity = 3.0       # Forward velocity of the robot
        self.robot_radius = 1.0          # Radius of robot for collision detection
        
        # Robot state
        self.robot_pose = np.zeros(3, dtype=np.float32)  # x, y, theta
        self.goal_position = np.array([24.0, -24.0], dtype=np.float32)
        
        # Lidar simulation parameters
        self.n_lidar_rays = 20
        self.max_lidar_distance = 30.0
        
        # Image representations
        self.image = np.zeros((stage_size, stage_size, 3), dtype=np.uint8)
        self.image_for_clash_calc = np.zeros((stage_size, stage_size), dtype=np.uint8)
        
        # Episode tracking
        self.prev_distance = 0.0
        self.timestep = 0
        self.max_episode_steps = 220
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Action and observation spaces
        # Action: Angular velocity (steering)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation: Lidar readings (normalized) + goal relative position
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(22,),  # 20 lidar readings + 2 for goal position (x, y)
            dtype=np.float32
        )
        
        # Regions for obstacle placement (7 regions x 3x3 grid = 63 possible positions)
        self.regions = [
            # Region 1: Top-left
            [
                [[-8, 32, -2.3], [0.0, 32, -2.3], [8, 32, -2.3]],
                [[-8, 24, -2.3], [0.0, 24, -2.3], [8, 24, -2.3]],
                [[-8, 16, -2.3], [0.0, 16, -2.3], [8, 16, -2.3]]
            ],
            # Region 2: Top-right
            [
                [[16, 32, -2.3], [24, 32, -2.3], [32, 32, -2.3]],
                [[16, 24, -2.3], [24, 24, -2.3], [32, 24, -2.3]],
                [[16, 16, -2.3], [24, 16, -2.3], [32, 16, -2.3]]
            ],
            # Region 3: Middle-left
            [
                [[-32, 8, -2.3], [-24, 8, -2.3], [-16, 8, -2.3]],
                [[-32, 0.0, -2.3], [-24, 0.0, -2.3], [-16, 0.0, -2.3]],
                [[-32, -8, -2.3], [-24, -8, -2.3], [-16, -8, -2.3]]
            ],
            # Region 4: Middle-center
            [
                [[-8, 8, -2.3], [0.0, 8, -2.3], [8, 8, -2.3]],
                [[-8, 0.0, -2.3], [0.0, 0.0, -2.3], [8, 0.0, -2.3]],
                [[-8, -8, -2.3], [0.0, -8, -2.3], [8, -8, -2.3]]
            ],
            # Region 5: Middle-right
            [
                [[16, 8, -2.3], [24, 8, -2.3], [32, 8, -2.3]],
                [[16, 0.0, -2.3], [24, 0.0, -2.3], [32, 0.0, -2.3]],
                [[16, -8, -2.3], [24, -8, -2.3], [32, -8, -2.3]]
            ],
            # Region 6: Bottom-left
            [
                [[-32, -16, -2.3], [-24, -16, -2.3], [-16, -16, -2.3]],
                [[-32, -24, -2.3], [-24, -24, -2.3], [-16, -24, -2.3]],
                [[-32, -32, -2.3], [-24, -32, -2.3], [-16, -32, -2.3]]
            ],
            # Region 7: Bottom-center
            [
                [[-8, -16, -2.3], [0.0, -16, -2.3], [8, -16, -2.3]],
                [[-8, -24, -2.3], [0.0, -24, -2.3], [8, -24, -2.3]],
                [[-8, -32, -2.3], [0.0, -32, -2.3], [8, -32, -2.3]]
            ]
        ]
        
        # Store obstacle positions
        self.obstacle_positions = []
        self.obstacle_list = []  # Will store obstacle data based on shape type
        self.boundary_margin = 5.0  # Distance from boundaries to consider collision
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state with random obstacle placement.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Initialize RNG
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.timestep = 0
        self.prev_distance = 0.0
        
        # Clear images (still needed for rendering)
        self.image.fill(0)
        self.image_for_clash_calc.fill(0)
        
        # Draw start and goal regions
        self._draw_start_goal_regions()
        
        # Generate random positions for obstacles
        self.obstacle_positions = self._generate_random_obstacle_positions()
        
        # Place obstacles in the environment - we'll store their positions
        self.obstacle_list = self._place_obstacles(self.obstacle_positions)
        
        # Add boundary walls (visual only, collision handled separately)
        self._add_boundary_walls()
        
        # Reset robot to starting position (farther from edges)
        self.robot_pose = np.array([-20.0, 20.0, 0.0], dtype=np.float32)
        
        # Calculate initial distance to goal
        pos_diff = np.array([self.robot_pose[0] - self.goal_position[0],
                             self.robot_pose[1] - self.goal_position[1]])
        self.prev_distance = np.linalg.norm(pos_diff)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (angular velocity)
            
        Returns:
            observation: New observation
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        self.timestep += 1
        
        # Scale action from [-1, 1] to realistic angular velocity range
        angular_velocity = 2.0 * action[0]
        
        # Update robot position using bicycle model
        self._update_robot_position(angular_velocity)
        
        # Check for collisions with the simpler method
        has_collision = self._check_collisions_simple()
        
        # Calculate distance to goal
        pos_diff = np.array([self.robot_pose[0] - self.goal_position[0],
                            self.robot_pose[1] - self.goal_position[1]])
        distance_to_goal = np.linalg.norm(pos_diff)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward and check termination
        reward, terminated = self._calculate_reward(distance_to_goal, has_collision)
        
        # Update previous distance for next step
        self.prev_distance = distance_to_goal
        
        # Check if we've reached max steps (truncation)
        truncated = self.timestep >= self.max_episode_steps
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, {}
    
    def _update_robot_position(self, angular_velocity):
        """
        Update the robot's position based on the bicycle model.
        
        Args:
            angular_velocity: Angular velocity for steering
        """
        # Time step for simulation
        dt = 0.1
        
        # Current robot state
        x, y, theta = self.robot_pose
        
        # Update position using simplified differential drive model
        theta_new = theta + angular_velocity * dt
        
        # Normalize angle to [-pi, pi]
        theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
        
        # Update position
        x_new = x + self.linear_velocity * dt * np.cos(theta_new)
        y_new = y + self.linear_velocity * dt * np.sin(theta_new)
        
        # Update robot pose
        self.robot_pose = np.array([x_new, y_new, theta_new], dtype=np.float32)
    
    def _check_collisions_simple(self):
        """
        Check collisions using a method appropriate for the obstacle shape.
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Check boundaries first (simple box collision)
        x, y, _ = self.robot_pose
        
        # Check if too close to boundaries (stage is 96x96 meters)
        if (x < -48 + self.boundary_margin or x > 48 - self.boundary_margin or
            y < -48 + self.boundary_margin or y > 48 - self.boundary_margin):
            return True
        
        # Check collisions based on obstacle shape
        if self.obstacle_shape == "circular":
            # For circular obstacles (circle-circle collision)
            for obs_x, obs_y, obs_radius in self.obstacle_list:
                # Calculate distance between robot and obstacle center
                distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                
                # Check if distance is less than sum of radii
                if distance < (self.robot_radius + obs_radius):
                    return True
        
        else:  # "square" obstacles
            # For square obstacles (circle-rectangle collision)
            robot_radius = self.robot_radius
            
            for obs_x, obs_y, half_width, half_height in self.obstacle_list:
                # Calculate the closest point on the rectangle to the circle center
                closest_x = max(obs_x - half_width, min(x, obs_x + half_width))
                closest_y = max(obs_y - half_height, min(y, obs_y + half_height))
                
                # Calculate distance between closest point and circle center
                distance = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
                
                # Check if distance is less than robot radius
                if distance < robot_radius:
                    return True
                
        return False
    
    def _get_observation(self):
        """
        Get the current observation (lidar readings + normalized goal position).
        
        Returns:
            observation: Normalized observation vector
        """
        # Simulate lidar readings
        lidar_data = self._simulate_lidar()
        
        # Create observation
        observation = np.zeros(22, dtype=np.float32)
        
        # Normalize lidar data
        observation[:20] = lidar_data / self.max_lidar_distance
        
        # Normalized goal position (relative to robot)
        observation[20:22] = np.array([
            -(self.robot_pose[0] - self.goal_position[0]) / 48,
            (self.robot_pose[1] - self.goal_position[1]) / 48
        ])
        
        return observation
    
    def _simulate_lidar(self):
        """
        Simulate LIDAR using ray-casting based on the obstacle shape.
        
        Returns:
            lidar_data: Array of distances from robot to obstacles
        """
        # Initialize lidar data
        lidar_data = np.full(self.n_lidar_rays, self.max_lidar_distance, dtype=np.float32)
        
        # Robot position
        x, y, theta = self.robot_pose
        
        # Cast rays in different directions
        for i in range(self.n_lidar_rays):
            # Calculate angle for this ray
            ray_angle = theta + (i * 2 * np.pi / self.n_lidar_rays)
            
            # Ray direction unit vector
            ray_dir_x = np.cos(ray_angle)
            ray_dir_y = np.sin(ray_angle)
            
            # Check distance to boundaries
            # For each boundary, calculate intersection with ray
            # Boundary equations: x = -48, x = 48, y = -48, y = 48
            
            # Left boundary: x = -48
            if ray_dir_x < 0:  # Ray pointing left
                t = (-48 - x) / ray_dir_x
                intersect_y = y + t * ray_dir_y
                if -48 <= intersect_y <= 48 and t > 0:
                    lidar_data[i] = min(lidar_data[i], t)
            
            # Right boundary: x = 48
            if ray_dir_x > 0:  # Ray pointing right
                t = (48 - x) / ray_dir_x
                intersect_y = y + t * ray_dir_y
                if -48 <= intersect_y <= 48 and t > 0:
                    lidar_data[i] = min(lidar_data[i], t)
            
            # Bottom boundary: y = -48
            if ray_dir_y < 0:  # Ray pointing down
                t = (-48 - y) / ray_dir_y
                intersect_x = x + t * ray_dir_x
                if -48 <= intersect_x <= 48 and t > 0:
                    lidar_data[i] = min(lidar_data[i], t)
            
            # Top boundary: y = 48
            if ray_dir_y > 0:  # Ray pointing up
                t = (48 - y) / ray_dir_y
                intersect_x = x + t * ray_dir_x
                if -48 <= intersect_x <= 48 and t > 0:
                    lidar_data[i] = min(lidar_data[i], t)
            
            # Check intersection with obstacles based on shape
            if self.obstacle_shape == "circular":
                # Circular obstacle intersection
                for obs_x, obs_y, obs_radius in self.obstacle_list:
                    # Vector from robot to obstacle center
                    to_center_x = obs_x - x
                    to_center_y = obs_y - y
                    
                    # Calculate dot product (projection of obstacle-center vector onto ray)
                    dot_product = to_center_x * ray_dir_x + to_center_y * ray_dir_y
                    
                    # If dot product is negative, obstacle is behind the ray
                    if dot_product < 0:
                        continue
                    
                    # Find closest point on ray to obstacle center
                    closest_x = x + dot_product * ray_dir_x
                    closest_y = y + dot_product * ray_dir_y
                    
                    # Distance from closest point to obstacle center
                    closest_distance = np.sqrt((closest_x - obs_x)**2 + (closest_y - obs_y)**2)
                    
                    # If this distance is greater than obstacle radius, ray doesn't hit
                    if closest_distance > obs_radius:
                        continue
                    
                    # Calculate distance from robot to intersection point
                    # Using Pythagorean theorem
                    intersection_distance = dot_product - np.sqrt(obs_radius**2 - closest_distance**2)
                    
                    # Update lidar reading if this is closer
                    if 0 < intersection_distance < lidar_data[i]:
                        lidar_data[i] = intersection_distance
            
            else:  # "square" obstacles
                # Square obstacle intersection
                for obs_x, obs_y, half_width, half_height in self.obstacle_list:
                    # Obstacle bounds
                    x_min = obs_x - half_width
                    x_max = obs_x + half_width
                    y_min = obs_y - half_height
                    y_max = obs_y + half_height
                    
                    # Check intersection with each side of the rectangle
                    # We need to check all four sides of the rectangle
                    
                    # Check intersection with bottom side (y = y_min)
                    if ray_dir_y != 0:  # Avoid division by zero
                        t_bottom = (y_min - y) / ray_dir_y
                        x_intersect = x + t_bottom * ray_dir_x
                        if t_bottom > 0 and x_min <= x_intersect <= x_max:
                            lidar_data[i] = min(lidar_data[i], t_bottom)
                    
                    # Check intersection with top side (y = y_max)
                    if ray_dir_y != 0:  # Avoid division by zero
                        t_top = (y_max - y) / ray_dir_y
                        x_intersect = x + t_top * ray_dir_x
                        if t_top > 0 and x_min <= x_intersect <= x_max:
                            lidar_data[i] = min(lidar_data[i], t_top)
                    
                    # Check intersection with left side (x = x_min)
                    if ray_dir_x != 0:  # Avoid division by zero
                        t_left = (x_min - x) / ray_dir_x
                        y_intersect = y + t_left * ray_dir_y
                        if t_left > 0 and y_min <= y_intersect <= y_max:
                            lidar_data[i] = min(lidar_data[i], t_left)
                    
                    # Check intersection with right side (x = x_max)
                    if ray_dir_x != 0:  # Avoid division by zero
                        t_right = (x_max - x) / ray_dir_x
                        y_intersect = y + t_right * ray_dir_y
                        if t_right > 0 and y_min <= y_intersect <= y_max:
                            lidar_data[i] = min(lidar_data[i], t_right)
        
        # Clamp values to max range
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_distance)
        
        return lidar_data
    
    def _calculate_reward(self, distance_to_goal, has_collision):
        """Modified reward function with stronger goal incentives and time penalty"""
        # Check for collision
        if has_collision:
            return -15, True  # Increased from -10 to -15
                
        # Check for goal reached - much higher reward
        at_goal = (20.0 < self.robot_pose[0] < 28.0 and 
                -28.0 < self.robot_pose[1] < -20.0)
        if at_goal:
            return 1000, True
                
        # Get minimum lidar reading
        lidar_data = self._simulate_lidar()
        min_lidar = np.min(lidar_data)
        
        # Penalty for getting too close to obstacles
        if min_lidar < 2:
            return -8, False  # Increased from -5 to -8
                
        # Base time penalty to discourage lengthy episodes
        time_penalty = -0.2  # Small constant penalty per timestep
        
        # Distance-based reward component
        if distance_to_goal < self.prev_distance:
            # Scaled by distance - more reward as agent gets closer to goal
            progress_scale = 1.5 * (1 - distance_to_goal / 67.22)**2  # Quadratic scaling
            progress_reward = 20 * progress_scale  # Increased base from 10 to 20
            
            # Add bonus for facing the goal - increased importance
            angle_to_goal = np.arctan2(self.goal_position[1] - self.robot_pose[1], 
                                    self.goal_position[0] - self.robot_pose[0])
            heading_diff = abs(angle_to_goal - self.robot_pose[2]) % (2 * np.pi)
            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
            
            # Sharper heading bonus that heavily rewards direct alignment
            heading_bonus = 10 * (1 - (heading_diff / np.pi)**2)  # Quadratic, max 10
            
            # Add distance threshold bonus to encourage getting close to goal
            distance_threshold_bonus = 0
            if distance_to_goal < 20:  # If within 20 units of goal
                distance_threshold_bonus = 15  # Extra incentive when getting close
            
            return time_penalty + progress_reward + max(0, heading_bonus) + distance_threshold_bonus, False
        else:
            # Larger penalty for moving away from goal
            moving_away_penalty = -2  # Increased from -1 to -2
            return time_penalty + moving_away_penalty, False
            
    def _generate_random_obstacle_positions(self):
        """
        Generate random positions for obstacles.
        
        Returns:
            positions: List of positions for obstacles
        """
        # For each of the 7 regions, generate 2 unique positions from 0-8
        positions = []
        for _ in range(7):
            # Generate 2 unique numbers in range 0-8
            region_positions = self.np_random.choice(9, size=2, replace=False)
            positions.append(region_positions)
            
        return positions
    
    def _place_obstacles(self, obstacle_positions):
        """
        Place obstacles in the environment based on random positions.
        
        Args:
            obstacle_positions: List of positions for obstacles
        
        Returns:
            obstacle_list: List of obstacle data based on shape type
        """
        obstacle_list = []
        
        # For each region, place two obstacles at random positions
        for region_idx, pos_pair in enumerate(obstacle_positions):
            for pos_idx in range(2):  # Two obstacles per region
                # Calculate position from region and random choice
                pos_value = pos_pair[pos_idx]
                row = pos_value // 3
                col = pos_value % 3
                coords = self.regions[region_idx][row][col]
                
                # Store obstacle data based on shape
                if self.obstacle_shape == "circular":
                    # For circular: [x, y, radius]
                    obstacle_list.append([coords[0], coords[1], self.obstacle_size/2])
                else:
                    # For square: [x, y, half_width, half_height]
                    half_size = self.obstacle_size/2
                    obstacle_list.append([coords[0], coords[1], half_size, half_size])
                
                # Calculate drawing coordinates (for visualization)
                pt1 = (int((self.stage_width/2 - self.obstacle_size/2 + coords[0])/self.pixel_to_meter), 
                      int((self.stage_height/2 + self.obstacle_size/2 - coords[1])/self.pixel_to_meter))
                pt2 = (int((self.stage_width/2 + self.obstacle_size/2 + coords[0])/self.pixel_to_meter), 
                      int((self.stage_height/2 - self.obstacle_size/2 - coords[1])/self.pixel_to_meter))
                
                # Draw obstacle on images (for visualization only)
                cv2.rectangle(self.image, pt1, pt2, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(self.image_for_clash_calc, pt1, pt2, 255, cv2.FILLED, cv2.LINE_8)
        
        return obstacle_list
    
    def _draw_start_goal_regions(self):
        """
        Draw start and goal regions on the image.
        """
        # Goal region (green)
        pt_start1 = (int((self.stage_width/2 - self.obstacle_size/2 + 24)/self.pixel_to_meter), 
                    int((self.stage_height/2 + self.obstacle_size/2 + 24)/self.pixel_to_meter))
        pt_start2 = (int((self.stage_width/2 + self.obstacle_size/2 + 24)/self.pixel_to_meter), 
                    int((self.stage_height/2 - self.obstacle_size/2 + 24)/self.pixel_to_meter))
        
        # Start region (red)
        pt_goal1 = (int((self.stage_width/2 - self.obstacle_size/2 - 24)/self.pixel_to_meter), 
                    int((self.stage_height/2 + self.obstacle_size/2 - 24)/self.pixel_to_meter))
        pt_goal2 = (int((self.stage_width/2 + self.obstacle_size/2 - 24)/self.pixel_to_meter), 
                    int((self.stage_height/2 - self.obstacle_size/2 - 24)/self.pixel_to_meter))
        
        # Draw rectangles
        cv2.rectangle(self.image, pt_start1, pt_start2, (0, 255, 0), cv2.FILLED, cv2.LINE_8)
        cv2.rectangle(self.image, pt_goal1, pt_goal2, (0, 0, 255), cv2.FILLED, cv2.LINE_8)
    
    def _add_boundary_walls(self):
        """
        Add boundary walls to the environment (for visualization only).
        """
        # Add walls with slice operations for efficiency
        self.image_for_clash_calc[0:4, :] = 255
        self.image_for_clash_calc[self.stage_size-4:self.stage_size, :] = 255
        self.image_for_clash_calc[:, 0:4] = 255
        self.image_for_clash_calc[:, self.stage_size-4:self.stage_size] = 255
        
        # Also draw walls on the rendered image
        self.image[0:4, :] = [100, 100, 100]
        self.image[self.stage_size-4:self.stage_size, :] = [100, 100, 100]
        self.image[:, 0:4] = [100, 100, 100]
        self.image[:, self.stage_size-4:self.stage_size] = [100, 100, 100]
    
    def _render_frame(self):
        """
        Render the current frame for visualization.
        Fixed rendering implementation for VectorizedDDEnv.
        """
        if self.render_mode == "human":
            try:
                import pygame
                
                # Initialize pygame
                if self.window is None:
                    pygame.init()
                    pygame.display.init()
                    self.window = pygame.display.set_mode((self.stage_size, self.stage_size))
                    pygame.display.set_caption("Vectorized DD Environment")
                
                if self.clock is None:
                    self.clock = pygame.time.Clock()
                
                # Clear screen with a light gray background
                self.window.fill((240, 240, 240))
                
                # FIXED: Consistent world_to_screen transformation for both render modes
                def world_to_screen(wx, wy):
                    # Convert from world coordinates to screen coordinates
                    sx = int(self.stage_size/2 + wx * self.stage_size/self.stage_width)
                    sy = int(self.stage_size/2 - wy * self.stage_size/self.stage_height)
                    return sx, sy
                
                # Draw boundaries
                pygame.draw.rect(self.window, (100, 100, 100), pygame.Rect(0, 0, self.stage_size, 4))  # Top
                pygame.draw.rect(self.window, (100, 100, 100), pygame.Rect(0, self.stage_size-4, self.stage_size, 4))  # Bottom
                pygame.draw.rect(self.window, (100, 100, 100), pygame.Rect(0, 0, 4, self.stage_size))  # Left
                pygame.draw.rect(self.window, (100, 100, 100), pygame.Rect(self.stage_size-4, 0, 4, self.stage_size))  # Right
                
                # Draw goal region (green square)
                goal_pos = world_to_screen(24.0, -24.0)
                goal_size = int(self.obstacle_size * self.stage_size/self.stage_width)
                pygame.draw.rect(self.window, (0, 255, 0), pygame.Rect(
                    goal_pos[0] - goal_size//2, 
                    goal_pos[1] - goal_size//2, 
                    goal_size, goal_size
                ))
                
                # Draw start region (red square)
                start_pos = world_to_screen(-24.0, 24.0)
                pygame.draw.rect(self.window, (255, 0, 0), pygame.Rect(
                    start_pos[0] - goal_size//2, 
                    start_pos[1] - goal_size//2, 
                    goal_size, goal_size
                ))
                
                # Draw obstacles based on shape
                if self.obstacle_shape == "circular":
                    # Draw circular obstacles
                    for obs_x, obs_y, obs_radius in self.obstacle_list:
                        # Convert to screen coordinates
                        obs_screen_x, obs_screen_y = world_to_screen(obs_x, obs_y)
                        # Scale radius to pixels
                        obs_screen_radius = int(obs_radius * self.stage_size/self.stage_width)
                        
                        # Draw obstacle with filled circle and border
                        pygame.draw.circle(self.window, (100, 100, 100), (obs_screen_x, obs_screen_y), obs_screen_radius)  # Darker gray
                        pygame.draw.circle(self.window, (0, 0, 0), (obs_screen_x, obs_screen_y), obs_screen_radius, 2)  # Black border, thicker
                
                else:  # "square" obstacles
                    # Draw square obstacles
                    for obs_x, obs_y, half_width, half_height in self.obstacle_list:
                        # Convert to screen coordinates
                        obs_screen_x, obs_screen_y = world_to_screen(obs_x, obs_y)
                        # Scale half width and height to pixels
                        half_width_pixels = int(half_width * self.stage_size/self.stage_width)
                        half_height_pixels = int(half_height * self.stage_size/self.stage_height)
                        
                        # Create rectangle for drawing
                        rect = pygame.Rect(
                            obs_screen_x - half_width_pixels,
                            obs_screen_y - half_height_pixels,
                            half_width_pixels * 2,
                            half_height_pixels * 2
                        )
                        
                        # Draw filled rectangle
                        pygame.draw.rect(self.window, (100, 100, 100), rect)
                        # Draw border
                        pygame.draw.rect(self.window, (0, 0, 0), rect, 2)
                
                # Draw robot
                robot_pos = world_to_screen(self.robot_pose[0], self.robot_pose[1])
                robot_radius = int(self.robot_radius * self.stage_size/self.stage_width)
                
                # Draw robot body
                pygame.draw.circle(self.window, (255, 0, 0), robot_pos, robot_radius)
                
                # Draw direction indicator
                theta = self.robot_pose[2]
                dir_length = robot_radius * 2
                dir_end = (
                    int(robot_pos[0] + dir_length * np.cos(theta)),
                    int(robot_pos[1] - dir_length * np.sin(theta))
                )
                pygame.draw.line(self.window, (0, 0, 255), robot_pos, dir_end, 2)
                
                # Draw lidar rays
                lidar_data = self._simulate_lidar()
                for i in range(len(lidar_data)):
                    ray_angle = self.robot_pose[2] + (i * 2 * np.pi / len(lidar_data))
                    ray_end_x = self.robot_pose[0] + lidar_data[i] * np.cos(ray_angle)
                    ray_end_y = self.robot_pose[1] + lidar_data[i] * np.sin(ray_angle)
                    
                    ray_start = world_to_screen(self.robot_pose[0], self.robot_pose[1])
                    ray_end = world_to_screen(ray_end_x, ray_end_y)
                    
                    pygame.draw.line(self.window, (0, 200, 200), ray_start, ray_end, 1)                
                
                # Draw text information
                try:
                    font = pygame.font.Font(None, 24)
                    
                    # Position
                    pos_text = font.render(f"Position: ({self.robot_pose[0]:.1f}, {self.robot_pose[1]:.1f})", True, (0, 0, 0))
                    self.window.blit(pos_text, (10, 10))
                    
                    # Heading
                    angle_text = font.render(f"Heading: {self.robot_pose[2]*180/np.pi:.1f}°", True, (0, 0, 0))
                    self.window.blit(angle_text, (10, 35))
                    
                    # Distance to goal
                    dist = np.linalg.norm(self.robot_pose[:2] - self.goal_position)
                    dist_text = font.render(f"Distance to goal: {dist:.1f}m", True, (0, 0, 0))
                    self.window.blit(dist_text, (10, 60))
                    
                    # Obstacle info
                    obstacle_text = font.render(f"Obstacles: {len(self.obstacle_list)} ({self.obstacle_shape})", True, (0, 0, 0))
                    self.window.blit(obstacle_text, (10, 85))
                except Exception as e:
                    print(f"Text rendering error: {e}")
                
                # Update display
                pygame.display.flip()
                
                # Control frame rate
                self.clock.tick(self.metadata["render_fps"])
                
                # Handle events (to prevent window from becoming unresponsive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        
            except Exception as e:
                print(f"Rendering error: {e}")
                import traceback
                traceback.print_exc()
                self.render_mode = None
        
        elif self.render_mode == "rgb_array":
            import pygame
            
            # Initialize pygame if needed
            if not pygame.get_init():
                pygame.init()
            
            # Create a surface for off-screen rendering
            surf = pygame.Surface((self.stage_size, self.stage_size))
            surf.fill((240, 240, 240))
            
            # FIXED: Consistent world_to_screen transformation for both render modes
            def world_to_screen(wx, wy):
                # Convert from world coordinates to screen coordinates
                sx = int(self.stage_size/2 + wx * self.stage_size/self.stage_width)
                sy = int(self.stage_size/2 - wy * self.stage_size/self.stage_height)
                return sx, sy
                
            # Draw boundaries
            pygame.draw.rect(surf, (100, 100, 100), pygame.Rect(0, 0, self.stage_size, 4))  # Top
            pygame.draw.rect(surf, (100, 100, 100), pygame.Rect(0, self.stage_size-4, self.stage_size, 4))  # Bottom
            pygame.draw.rect(surf, (100, 100, 100), pygame.Rect(0, 0, 4, self.stage_size))  # Left
            pygame.draw.rect(surf, (100, 100, 100), pygame.Rect(self.stage_size-4, 0, 4, self.stage_size))  # Right
            
            # Draw goal region (green square)
            goal_pos = world_to_screen(24.0, -24.0)
            goal_size = int(self.obstacle_size * self.stage_size/self.stage_width)
            pygame.draw.rect(surf, (0, 180, 0), pygame.Rect(
                goal_pos[0] - goal_size//2, 
                goal_pos[1] - goal_size//2, 
                goal_size, goal_size
            ))
            
            # Draw start region (red square)
            start_pos = world_to_screen(-24.0, 24.0)
            pygame.draw.rect(surf, (180, 0, 0), pygame.Rect(
                start_pos[0] - goal_size//2, 
                start_pos[1] - goal_size//2, 
                goal_size, goal_size
            ))
            
            # Draw obstacles based on shape
            if self.obstacle_shape == "circular":
                # Draw circular obstacles
                for obs_x, obs_y, obs_radius in self.obstacle_list:
                    # Convert to screen coordinates
                    obs_screen_x, obs_screen_y = world_to_screen(obs_x, obs_y)
                    obs_screen_radius = int(obs_radius * self.stage_size/self.stage_width)
                    
                    # Draw obstacle
                    pygame.draw.circle(surf, (180, 180, 180), (obs_screen_x, obs_screen_y), obs_screen_radius)
                    pygame.draw.circle(surf, (100, 100, 100), (obs_screen_x, obs_screen_y), obs_screen_radius, 1)
            
            else:  # "square" obstacles
                # Draw square obstacles
                for obs_x, obs_y, half_width, half_height in self.obstacle_list:
                    # Convert to screen coordinates
                    obs_screen_x, obs_screen_y = world_to_screen(obs_x, obs_y)
                    # Scale half width and height to pixels
                    half_width_pixels = int(half_width * self.stage_size/self.stage_width)
                    half_height_pixels = int(half_height * self.stage_size/self.stage_height)
                    
                    # Create rectangle for drawing
                    rect = pygame.Rect(
                        obs_screen_x - half_width_pixels,
                        obs_screen_y - half_height_pixels,
                        half_width_pixels * 2,
                        half_height_pixels * 2
                    )
                    
                    # Draw filled rectangle
                    pygame.draw.rect(surf, (180, 180, 180), rect)
                    # Draw border
                    pygame.draw.rect(surf, (100, 100, 100), rect, 1)
            
            # Draw robot
            robot_pos = world_to_screen(self.robot_pose[0], self.robot_pose[1])
            robot_radius = int(self.robot_radius * self.stage_size/self.stage_width)
            
            # Draw robot body
            pygame.draw.circle(surf, (255, 0, 0), robot_pos, robot_radius)
            
            # Draw direction indicator
            theta = self.robot_pose[2]
            dir_length = robot_radius * 2
            dir_end = (
                int(robot_pos[0] + dir_length * np.cos(theta)),
                int(robot_pos[1] - dir_length * np.sin(theta))
            )
            pygame.draw.line(surf, (0, 0, 255), robot_pos, dir_end, 2)
            
            # Draw lidar rays
            lidar_data = self._simulate_lidar()
            for i in range(len(lidar_data)):
                ray_angle = self.robot_pose[2] + (i * 2 * np.pi / len(lidar_data))
                ray_end_x = self.robot_pose[0] + lidar_data[i] * np.cos(ray_angle)
                ray_end_y = self.robot_pose[1] + lidar_data[i] * np.sin(ray_angle)
                
                ray_start = world_to_screen(self.robot_pose[0], self.robot_pose[1])
                ray_end = world_to_screen(ray_end_x, ray_end_y)
                
                pygame.draw.line(surf, (0, 200, 200), ray_start, ray_end, 1)
            
            # ADDED: Draw center marker for debugging
            pygame.draw.circle(surf, (255, 0, 255), (self.stage_size//2, self.stage_size//2), 5)
            
            # Convert the surface to a numpy array
            arr = pygame.surfarray.array3d(surf)
            arr = np.transpose(arr, (1, 0, 2))
            return arr        
        return None
        
    def render(self):
        """
        Render the environment.
        
        Returns:
            rgb_array if render_mode is "rgb_array", else None
        """
        return self._render_frame()
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
        
        if hasattr(self, 'image'):
            del self.image
        if hasattr(self, 'image_for_clash_calc'):
            del self.image_for_clash_calc


# For the vectorized environment implementation, we need a vectorized wrapper
def make_vectorized_env(num_envs, seed=None, obstacle_shape="circular"):
    """
    Create a vectorized environment for parallel training.
    
    Args:
        num_envs: Number of environments to run in parallel
        seed: Random seed for reproducibility
        obstacle_shape: Shape of obstacles ('circular' or 'square')
        
    Returns:
        envs: Vectorized environment
    """
    from gymnasium.vector import SyncVectorEnv
    
    def make_env(index):
        def _init():
            env = VectorizedDDEnv(obstacle_shape=obstacle_shape)
            env.reset(seed=seed + index if seed is not None else None)
            return env
        return _init
    
    return SyncVectorEnv([make_env(i) for i in range(num_envs)])