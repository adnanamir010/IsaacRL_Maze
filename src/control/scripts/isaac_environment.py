import numpy as np
import math
import cv2
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from omni.isaac.core.prims import XFormPrim
from pxr import UsdGeom
import global_vars
import os
import weakref

class DDEnv(Node):
    """
    Memory-optimized robot environment for Differential Drive in Isaac Sim.
    This environment simulates a robot navigating through obstacles to reach a goal.
    """
    def __init__(self, world, simulation_context, image_size, pixel_to_meter,
                 stage_width, stage_height, wheelbase, img, img_for_clash_calc):
        """
        Initialize the robot environment with explicit dependencies.
        """
        super().__init__('environment')
        
        # Store passed references - use weakref for objects that may have circular references
        self.world = weakref.ref(world)
        self.simulation_context = weakref.ref(simulation_context)
        self.stage_width = stage_width
        self.stage_height = stage_height
        self.wheelbase = wheelbase
        self.pixel_to_meter = pixel_to_meter
        
        # Initialize global variables - create memory-mapped arrays
        self._setup_memmapped_images(image_size, img, img_for_clash_calc)

        # Robot control parameters
        self.steering_angle = np.zeros(2, dtype=np.float32)  # left, right knuckle positions
        self.wheel_speed = np.zeros(2, dtype=np.float32)     # left, right wheel velocities
        
        # Publishers for robot control
        self.steering_publisher = self.create_publisher(Float64MultiArray, '/controller/cmd_pos', 10)
        self.velocity_publisher = self.create_publisher(Float64MultiArray, '/controller/cmd_vel', 10)

        # Robot physical parameters
        self.track_width = 1.1   # Track width between wheels
        self.wheelbase = 1.8     # Wheelbase length
        self.wheel_radius = 0.3  # Wheel radius
        
        # State representation
        self.current_state = np.zeros(22, dtype=np.float32)  # 0-19: lidar readings, 20-21: goal distance (x,y)
        
        # Environment parameters
        self.goal_position = np.array([24.0, -24.0], dtype=np.float32)
        self.linear_velocity = 3.0     # Forward velocity of the robot
        self.angular_velocity = 0.0    # Angular velocity of the robot
        self.obstacle_size = 8.0
        self.prev_distance = 0.0       # Previous distance to goal for reward calculation
        self.done = False
        
        # Create obstacles (cubes) - store references efficiently
        self.cube_transforms = self._create_cube_transforms()
        self._initialize_cube_geometry()
    
    def _setup_memmapped_images(self, image_size, img, img_for_clash_calc):
        """Create memory-mapped arrays for images instead of in-memory arrays"""
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        # Define memmap files
        image_file = 'temp/environment_image.dat'
        clash_image_file = 'temp/clash_image.dat'
        
        # If passed existing images, we'll copy their data
        has_existing_data = (img is not None and img_for_clash_calc is not None)
        
        # Create memory-mapped arrays
        try:
            if has_existing_data:
                # Copy existing data
                image = np.memmap(image_file, dtype=np.uint8, mode='w+', 
                                 shape=(image_size, image_size, 3))
                image_for_clash_calc = np.memmap(clash_image_file, dtype=np.uint8, mode='w+',
                                              shape=(image_size, image_size))
                
                # Copy data if available
                np.copyto(image, img)
                np.copyto(image_for_clash_calc, img_for_clash_calc)
            else:
                # Create new empty arrays
                image = np.memmap(image_file, dtype=np.uint8, mode='w+', 
                                 shape=(image_size, image_size, 3))
                image_for_clash_calc = np.memmap(clash_image_file, dtype=np.uint8, mode='w+',
                                              shape=(image_size, image_size))
                
                # Initialize with zeros
                image.fill(0)
                image_for_clash_calc.fill(0)
                
            # Update global variables
            global_vars.image = image
            global_vars.image_for_clash_calc = image_for_clash_calc
            
            # Store references locally
            self.image_file = image_file
            self.clash_image_file = clash_image_file
            
        except Exception as e:
            self.get_logger().error(f"Error creating memory-mapped arrays: {e}")
            # Fall back to regular NumPy arrays if memmapping fails
            if not has_existing_data:
                global_vars.image = np.zeros((image_size, image_size, 3), np.uint8)
                global_vars.image_for_clash_calc = np.zeros((image_size, image_size), np.uint8)
    
    def _create_cube_transforms(self):
        """Create transform objects for all obstacle cubes"""
        transforms = []
        for i in range(1, 15):  # 14 cubes
            transforms.append(XFormPrim(prim_path=f"/World/Cube{i}"))
        return transforms
    
    def _initialize_cube_geometry(self):
        """Initialize the cube geometries for obstacles"""
        world_ref = self.world()
        if world_ref is None:
            self.get_logger().error("World reference is no longer valid")
            return
            
        stage = world_ref.stage
        
        for i, transform in enumerate(self.cube_transforms):
            # Create cube geometry
            cube_index = i + 1
            cube_geom = UsdGeom.Cube.Define(stage, f"/World/Cube{cube_index}/Geom")
            # Set cube size
            cube_geom.GetSizeAttr().Set(self.obstacle_size)

    def step(self, action, time_steps, max_episode_steps):
        """
        Execute one step in the environment with memory optimizations.
        """
        simulation_context_ref = self.simulation_context()
        if simulation_context_ref is None:
            self.get_logger().error("Simulation context reference is no longer valid")
            return self.current_state, -10, True
            
        self.done = False
        self.angular_velocity = 2 * action[0]

        # Calculate steering angles based on bicycle model - vectorized calculation
        denominators = np.array([
            2 * self.linear_velocity - self.angular_velocity * self.track_width,  # left
            2 * self.linear_velocity + self.angular_velocity * self.track_width   # right
        ])
        
        # Avoid division by zero
        mask = (denominators != 0)
        self.steering_angle = np.zeros(2, dtype=np.float32)
        self.steering_angle[mask] = np.arctan(self.angular_velocity * self.wheelbase / denominators[mask])

        # Calculate wheel velocities - vectorized
        self.wheel_speed = np.array([
            (self.linear_velocity - self.angular_velocity * self.track_width / 2) / self.wheel_radius,
            (self.linear_velocity + self.angular_velocity * self.track_width / 2) / self.wheel_radius
        ], dtype=np.float32)

        # Publish control commands
        wheel_speed_msg = Float64MultiArray(data=self.wheel_speed)    
        self.velocity_publisher.publish(wheel_speed_msg)  
        steering_angle_msg = Float64MultiArray(data=self.steering_angle)    
        self.steering_publisher.publish(steering_angle_msg)  

        # Simulate for multiple steps
        for _ in range(10):
            simulation_context_ref.step(render=True)

        # Calculate distance to goal - optimized with numpy
        pos_diff = np.array([global_vars.body_pose[0] - self.goal_position[0],
                             global_vars.body_pose[1] - self.goal_position[1]])
        distance_to_goal = np.linalg.norm(pos_diff)

        # Update state - vectorized operations
        self.current_state[:20] = global_vars.lidar_data / 30  # Normalize lidar data
        self.current_state[20:22] = np.array([
            -(global_vars.body_pose[0] - self.goal_position[0]) / 48,  # Normalized x distance
            (global_vars.body_pose[1] - self.goal_position[1]) / 48    # Normalized y distance
        ])

        # Calculate reward and check termination conditions
        reward = self._calculate_reward(distance_to_goal, global_vars.clash_sum, time_steps, max_episode_steps)
        
        # Update previous distance for next step
        self.prev_distance = distance_to_goal

        self.get_logger().info(f"State: {self.current_state}, Collision: {global_vars.clash_sum}, " 
                              f"Reward: {reward}, Angular velocity: {self.angular_velocity}")

        return self.current_state, reward, self.done

    def _calculate_reward(self, distance_to_goal, clash_sum, time_steps, max_episode_steps):
        """Modified reward function to match the current project's structure"""
        # Check for collision
        if clash_sum > 0:
            self.get_logger().info("COLLISION DETECTED. EPISODE ENDED.")
            self.done = True
            return -15 
                
        # Check for goal reached - much higher reward
        at_goal = (20.0 < global_vars.body_pose[0] < 28.0 and 
                -28.0 < global_vars.body_pose[1] < -20.0)
        if at_goal:
            self.get_logger().info("GOAL REACHED. EPISODE ENDED.")
            self.done = True
            return 1000
                
        # Get minimum lidar reading
        min_lidar = np.min(global_vars.lidar_data)
        
        # Penalty for getting too close to obstacles
        if min_lidar < 2:
            self.get_logger().info("TOO CLOSE TO OBSTACLE.")
            return -8
                
        # Base time penalty to discourage lengthy episodes
        time_penalty = -0.2
        
        # Distance-based reward component
        if distance_to_goal < self.prev_distance:
            # Scaled by distance - more reward as agent gets closer to goal
            progress_scale = 1.5 * (1 - distance_to_goal / 67.22)**2  # Quadratic scaling
            progress_reward = 20 * progress_scale  # Increased base from 10 to 20
            
            # Add bonus for facing the goal - increased importance
            angle_to_goal = np.arctan2(self.goal_position[1] - global_vars.body_pose[1], 
                                self.goal_position[0] - global_vars.body_pose[0])
            heading_diff = abs(angle_to_goal - global_vars.body_pose[2]) % (2 * np.pi)
            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
            
            # Sharper heading bonus that heavily rewards direct alignment
            heading_bonus = 10 * (1 - (heading_diff / np.pi)**2)  # Quadratic, max 10
            
            # Add distance threshold bonus to encourage getting close to goal
            distance_threshold_bonus = 0
            if distance_to_goal < 20:  # If within 20 units of goal
                distance_threshold_bonus = 15  # Extra incentive when getting close
            
            return time_penalty + progress_reward + max(0, heading_bonus) + distance_threshold_bonus
        else:
            # Larger penalty for moving away from goal
            moving_away_penalty = -2  # Increased from -1 to -2
            return time_penalty + moving_away_penalty
    
    def reset(self, euler_angle=None):
        """
        Reset the environment to initial state with random obstacle placement.
        Memory-optimized implementation.
        """
        simulation_context_ref = self.simulation_context()
        if simulation_context_ref is None:
            self.get_logger().error("Simulation context reference is no longer valid")
            return self.current_state
            
        # Reset simulation
        simulation_context_ref.reset()
        
        # Clear images efficiently
        if global_vars.image_for_clash_calc is not None:
            global_vars.image_for_clash_calc.fill(0)
            # Flush changes to disk if using memmapped array
            if hasattr(global_vars.image_for_clash_calc, 'flush'):
                global_vars.image_for_clash_calc.flush()
                
        if global_vars.image is not None:
            global_vars.image.fill(0)
            # Flush changes to disk if using memmapped array
            if hasattr(global_vars.image, 'flush'):
                global_vars.image.flush()
            
        self.prev_distance = 0
        global_vars.clash_sum = 0  # Reset collision counter

        # Draw start and goal regions
        self._draw_start_goal_regions()
        
        # Generate random positions for obstacles
        obstacle_positions = self._generate_random_obstacle_positions()
        
        # Place obstacles in the environment
        self._place_obstacles(obstacle_positions)
        
        # Add boundary walls
        self._add_boundary_walls()
        
        self.done = False
        self.current_state.fill(0.0)

        return self.current_state
        
    def _draw_start_goal_regions(self):
        """Draw start and goal regions on the image - optimized implementation"""
        # Pre-calculate coordinates for efficiency
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
        cv2.rectangle(global_vars.image, pt_start1, pt_start2, (0, 255, 0), cv2.FILLED, cv2.LINE_8)
        cv2.rectangle(global_vars.image, pt_goal1, pt_goal2, (0, 0, 255), cv2.FILLED, cv2.LINE_8)
        
        # Flush changes if using memmapped array
        if hasattr(global_vars.image, 'flush'):
            global_vars.image.flush()
    
    def _generate_random_obstacle_positions(self):
        """Generate random positions for obstacles - FIXED implementation"""
        # FIX: Generate individual positions for each pair to ensure enough unique positions
        rng = np.random.default_rng()
        positions = []
        
        # For each of the 7 regions, generate 2 unique positions from 0-8
        for _ in range(7):
            # Generate 2 unique numbers in range 0-8
            region_positions = rng.choice(9, size=2, replace=False)
            positions.append(region_positions)
            
        return positions
    
    def _place_obstacles(self, obstacle_positions):
        """Place obstacles in the environment based on random positions - optimized implementation"""
        # Predefined obstacle regions and their coordinates - use NumPy array for efficiency
        regions = np.array([
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
            # Regions 3-7 (omitted for brevity)
            # ...
        ], dtype=np.float32)
        
        # More regions (copied from original)
        regions = [
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
        
        # Pre-calculate obstacle drawing coordinates
        pt_cache = {}
        
        # For each region, place two obstacles at random positions
        cube_index = 0
        for region_idx, pos_pair in enumerate(obstacle_positions):
            for pos_idx in range(2):  # Two obstacles per region
                # Calculate position from region and random choice
                pos_value = pos_pair[pos_idx]
                row = pos_value // 3
                col = pos_value % 3
                coords = regions[region_idx][row][col]
                
                # Set obstacle position
                self.cube_transforms[cube_index].set_world_pose(coords, [0.0, 0.0, 0.0, 1.0])
                
                # Calculate drawing coordinates (cached for efficiency)
                key = (coords[0], coords[1])
                if key not in pt_cache:
                    pt1 = (int((self.stage_width/2 - self.obstacle_size/2 + coords[0])/self.pixel_to_meter), 
                          int((self.stage_height/2 + self.obstacle_size/2 - coords[1])/self.pixel_to_meter))
                    pt2 = (int((self.stage_width/2 + self.obstacle_size/2 + coords[0])/self.pixel_to_meter), 
                          int((self.stage_height/2 - self.obstacle_size/2 - coords[1])/self.pixel_to_meter))
                    pt_cache[key] = (pt1, pt2)
                else:
                    pt1, pt2 = pt_cache[key]
                
                # Draw obstacle on images
                cv2.rectangle(global_vars.image, pt1, pt2, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(global_vars.image_for_clash_calc, pt1, pt2, 255, cv2.FILLED, cv2.LINE_8)
                
                cube_index += 1
                
        # Flush memory-mapped changes if applicable
        if hasattr(global_vars.image, 'flush'):
            global_vars.image.flush()
        if hasattr(global_vars.image_for_clash_calc, 'flush'):
            global_vars.image_for_clash_calc.flush()

    def _add_boundary_walls(self):
        """Add boundary walls to the environment - memory-optimized implementation"""
        # Add walls with slice operations for efficiency
        global_vars.image_for_clash_calc[0:4, :] = 255
        global_vars.image_for_clash_calc[716:720, :] = 255
        global_vars.image_for_clash_calc[:, 0:4] = 255
        global_vars.image_for_clash_calc[:, 716:720] = 255
        
        # Flush memory-mapped changes if applicable
        if hasattr(global_vars.image_for_clash_calc, 'flush'):
            global_vars.image_for_clash_calc.flush()
            
    def __del__(self):
        """Clean up resources when object is deleted"""
        try:
            # Clean up memory-mapped files
            if hasattr(self, 'image_file') and os.path.exists(self.image_file):
                os.remove(self.image_file)
            if hasattr(self, 'clash_image_file') and os.path.exists(self.clash_image_file):
                os.remove(self.clash_image_file)
        except Exception as e:
            print(f"Error cleaning up environment resources: {e}")