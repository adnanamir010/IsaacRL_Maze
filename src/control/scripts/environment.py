import numpy as np
import math
import cv2
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from omni.isaac.core.objects import XFormPrim
from pxr import UsdGeom

class DDEnvironment(Node):
    """
    Robot environment for reinforcement learning in Isaac Sim.
    This environment simulates a robot navigating through obstacles to reach a goal.
    """
    def __init__(self):
        super().__init__('environment')

        # Robot control parameters
        self.steering_angle = np.array([0, 0], float)  # left, right knuckle positions
        self.wheel_speed = np.array([0, 0], float)     # left, right wheel velocities
        
        # Publishers for robot control
        self.steering_publisher = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.velocity_publisher = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)

        # Robot physical parameters
        self.track_width = 1.1   # Track width between wheels
        self.wheelbase = 1.8     # Wheelbase length
        self.wheel_radius = 0.3  # Wheel radius
        
        # State representation
        self.current_state = np.zeros(22)  # 0-19: lidar readings, 20-21: goal distance (x,y)
        
        # Environment parameters
        self.goal_position = [24.0, -24.0]
        self.linear_velocity = 3.0     # Forward velocity of the robot
        self.angular_velocity = 0.0    # Angular velocity of the robot
        self.obstacle_size = 8.0
        self.prev_distance = 0        # Previous distance to goal for reward calculation
        
        # Create obstacles (cubes)
        self.cube_transforms = self._create_cube_transforms()
        self._initialize_cube_geometry()
    
    def _create_cube_transforms(self):
        """Create transform objects for all obstacle cubes"""
        transforms = []
        for i in range(1, 15):  # 14 cubes
            transforms.append(XFormPrim(prim_path=f"/World/Cube{i}"))
        return transforms
    
    def _initialize_cube_geometry(self):
        """Initialize the cube geometries for obstacles"""
        stage = my_world.stage
        
        for i, transform in enumerate(self.cube_transforms):
            # Create cube geometry
            cube_index = i + 1
            cube_geom = UsdGeom.Cube.Define(stage, f"/World/Cube{cube_index}/Geom")
            # Set cube size
            cube_geom.GetSizeAttr().Set(self.obstacle_size)

    def step(self, action, time_steps, max_episode_steps):
        """
        Execute one step in the environment.
        
        Args:
            action: Control action for the robot (steering)
            time_steps: Current time step in the episode
            max_episode_steps: Maximum time steps per episode
            
        Returns:
            next_state: New state after taking the action
            reward: Reward for the action
            done: Whether the episode is finished
        """
        global body_pose, lidar_data, clash_sum

        self.done = False
        self.angular_velocity = 2 * action[0]

        # Calculate steering angles based on bicycle model
        left_denom = 2 * self.linear_velocity - self.angular_velocity * self.track_width
        right_denom = 2 * self.linear_velocity + self.angular_velocity * self.track_width
        
        # Calculate left steering angle
        if left_denom != 0:
            self.steering_angle[0] = math.atan(self.angular_velocity * self.wheelbase / left_denom)
        else:
            self.steering_angle[0] = 0
        
        # Calculate right steering angle
        if right_denom != 0:
            self.steering_angle[1] = math.atan(self.angular_velocity * self.wheelbase / right_denom)
        else:
            self.steering_angle[1] = 0

        # Calculate wheel velocities
        self.wheel_speed[0] = (self.linear_velocity - self.angular_velocity * self.track_width / 2) / self.wheel_radius
        self.wheel_speed[1] = (self.linear_velocity + self.angular_velocity * self.track_width / 2) / self.wheel_radius

        # Publish control commands
        wheel_speed_msg = Float64MultiArray(data=self.wheel_speed)    
        self.velocity_publisher.publish(wheel_speed_msg)  
        steering_angle_msg = Float64MultiArray(data=self.steering_angle)    
        self.steering_publisher.publish(steering_angle_msg)  

        # Simulate for multiple steps
        for _ in range(20):
            simulation_context.step(render=True)

        # Calculate distance to goal
        distance_to_goal = math.sqrt((body_pose[0] - self.goal_position[0])**2 + 
                                     (body_pose[1] - self.goal_position[1])**2)

        # Update state
        self.current_state[:20] = lidar_data / 30  # Normalize lidar data
        self.current_state[20] = -(body_pose[0] - self.goal_position[0]) / 48  # Normalized x distance
        self.current_state[21] = (body_pose[1] - self.goal_position[1]) / 48   # Normalized y distance

        # Calculate reward and check termination conditions
        reward = self._calculate_reward(distance_to_goal, clash_sum, time_steps, max_episode_steps)
        
        # Update previous distance for next step
        self.prev_distance = distance_to_goal

        self.get_logger().info(f"State: {self.current_state}, Collision: {clash_sum}, " 
                              f"Reward: {reward}, Angular velocity: {self.angular_velocity}")

        return self.current_state, reward, self.done

    def _calculate_reward(self, distance_to_goal, clash_sum, time_steps, max_episode_steps):
        """Calculate reward based on current state"""
        
        # Check for collision
        if clash_sum > 0:
            self.get_logger().info("COLLISION DETECTED. EPISODE ENDED.")
            self.done = True
            return -10
            
        # Check for timeout
        if time_steps >= max_episode_steps:
            self.get_logger().info("TIME LIMIT REACHED. EPISODE ENDED.")
            self.done = True
            return 0
            
        # Check for goal reached
        if (20.0 < body_pose[0] < 28.0 and -28.0 < body_pose[1] < -20.0):
            self.get_logger().info("GOAL REACHED. EPISODE ENDED.")
            self.done = True
            return 20
            
        # Penalty for getting too close to obstacles
        if min(lidar_data) < 3:
            self.get_logger().info("TOO CLOSE TO OBSTACLE.")
            return -5
            
        # Reward for making progress toward the goal
        if distance_to_goal < self.prev_distance:
            return 10 * max(1 - distance_to_goal / 67.22, 0)
        else:
            return -1  # Penalty for moving away from the goal

    def reset(self):
        """Reset the environment to initial state with random obstacle placement"""
        global euler_angle
        
        # Reset simulation and images
        simulation_context.reset()
        image_for_clash_calc[:,:] = 0
        image[:,:,:] = 0
        self.prev_distance = 0

        # Draw start and goal regions
        self._draw_start_goal_regions()
        
        # Generate random positions for obstacles
        obstacle_positions = self._generate_random_obstacle_positions()
        
        # Place obstacles in the environment
        self._place_obstacles(obstacle_positions)
        
        # Add boundary walls
        self._add_boundary_walls()
        
        self.done = False
        self.current_state[:] = 0.0

        return self.current_state
        
    def _draw_start_goal_regions(self):
        """Draw start and goal regions on the image"""
        # Start region (red)
        pt_start1 = (int((stage_W/2 - L/2 + 24)/pix2m), int((stage_H/2 + L/2 + 24)/pix2m))
        pt_start2 = (int((stage_W/2 + L/2 + 24)/pix2m), int((stage_H/2 - L/2 + 24)/pix2m))
        cv2.rectangle(image, pt_start1, pt_start2, (0, 0, 255), cv2.FILLED, cv2.LINE_8)
        
        # Goal region (blue)
        pt_goal1 = (int((stage_W/2 - L/2 - 24)/pix2m), int((stage_H/2 + L/2 - 24)/pix2m))
        pt_goal2 = (int((stage_W/2 + L/2 - 24)/pix2m), int((stage_H/2 - L/2 - 24)/pix2m))
        cv2.rectangle(image, pt_goal1, pt_goal2, (255, 0, 0), cv2.FILLED, cv2.LINE_8)
    
    def _generate_random_obstacle_positions(self):
        """Generate random positions for obstacles"""
        rng = np.random.default_rng()
        positions = []
        for _ in range(7):  # 7 pairs of obstacles
            numbers = rng.choice(9, size=2, replace=False)
            positions.append(numbers)
        return positions
    
    def _place_obstacles(self, obstacle_positions):
        """Place obstacles in the environment based on random positions"""
        # Predefined obstacle regions and their coordinates
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
                
                # Draw obstacle on image
                pt1 = (int((stage_W/2 - L/2 + coords[0])/pix2m), int((stage_H/2 + L/2 - coords[1])/pix2m))
                pt2 = (int((stage_W/2 + L/2 + coords[0])/pix2m), int((stage_H/2 - L/2 - coords[1])/pix2m))
                cv2.rectangle(image, pt1, pt2, (200, 200, 200), cv2.FILLED, cv2.LINE_8)
                cv2.rectangle(image_for_clash_calc, pt1, pt2, 255, cv2.FILLED, cv2.LINE_8)
                
                cube_index += 1
    
    def _add_boundary_walls(self):
        """Add boundary walls to the environment"""
        image_for_clash_calc[0:4, :] = 255
        image_for_clash_calc[716:720, :] = 255
        image_for_clash_calc[:, 0:4] = 255
        image_for_clash_calc[:, 716:720] = 255