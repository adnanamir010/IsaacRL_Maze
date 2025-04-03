#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import threading
import numpy as np
import math
import cv2
import itertools
import torch
import argparse
import datetime
import copy
import os
import matplotlib.pyplot as plt

# Initialize Isaac Sim Application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# Import Isaac Sim modules
import omni
from omni.isaac.core import World

# Import required modules
from agents import SAC
from memory import ReplayMemory
from environment import DDEnv
from rl_utils import ModelStateSubscriber, LidarSubscriber, CollisionDetector, save_learning_curve
from torch.utils.tensorboard import SummaryWriter
import global_vars

# Enable ROS2 bridge extension
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

# Spawn world with physics settings
my_world = World(stage_units_in_meters=1.0, physics_dt=1/200, rendering_dt=1/20)

# Load the USD stage
isaac_path = os.environ["HOME"] + "/projects/IsaacRL_Maze/src/stage.usd"
omni.usd.get_context().open_stage(isaac_path)

# Wait for things to load
simulation_app.update()
while omni.isaac.core.utils.stage.is_stage_loading():
    simulation_app.update()

# Initialize simulation context
simulation_context = omni.isaac.core.SimulationContext()
simulation_context.initialize_physics()
simulation_context.play()

# Global variables for robot state and environment
body_pose = np.array([0.0, 0.0, 0.0], float)  # x, y, theta
lidar_data = np.zeros(20)
clash_sum = 0

# Image settings for visualization and collision detection
image_size = 720
pixel_to_meter = 0.1  # m/pix
wheelbase = 8  # length of robot
stage_width = image_size * pixel_to_meter
stage_height = image_size * pixel_to_meter

# Create images for visualization and collision detection
image = np.zeros((image_size, image_size, 3), np.uint8)
image_for_clash_calc = np.zeros((image_size, image_size), np.uint8)

# Set numpy print format for better readability
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

def parse_arguments():
    """Parse command line arguments for the SAC algorithm"""
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Add your arguments here (copied from your code)
    parser.add_argument('--env-name', default="obstacle_avoidance",
                    help='Environment name (default: obstacle_avoidance)')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluate policy every 10 episodes (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='Target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='Learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α for entropy (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='Batch size (default: 64)')
    parser.add_argument('--num_steps', type=int, default=300_001, metavar='N',
                    help='Maximum number of steps (default: 500_001)')
    parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                    help='Hidden size (default: 64)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='Model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=4000, metavar='N',
                    help='Steps sampling random actions (default: 4000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update interval (default: 1)')
    parser.add_argument('--replay_size', type=int, default=200000, metavar='N',
                    help='Size of replay buffer (default: 200000)')
    parser.add_argument('--cuda', action="store_true",
                    help='Run on CUDA (default: False)')
    parser.add_argument('--save-curve', action="store_true",
                    help='Save learning curve plot (default: False)')
    parser.add_argument('--curve-name', type=str, default='learning_curve',
                    help='Filename for learning curve plot (default: learning_curve)')
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize ROS2
    rclpy.init(args=None)
    
    # Create images for visualization and collision detection
    global_vars.image = np.zeros((image_size, image_size, 3), np.uint8)
    global_vars.image_for_clash_calc = np.zeros((image_size, image_size), np.uint8)
    
    # Initialize environment and subscribers
    env = DDEnv(my_world, simulation_context, image_size, pixel_to_meter,
               stage_width, stage_height, wheelbase, global_vars.image, global_vars.image_for_clash_calc)
    
    model_state_subscriber = ModelStateSubscriber()
    lidar_subscriber = LidarSubscriber()
    collision_detector = CollisionDetector(image_size, pixel_to_meter)
    
    # Set up ROS2 executor for multi-threading
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env)
    executor.add_node(model_state_subscriber)
    executor.add_node(lidar_subscriber)
    executor.add_node(collision_detector)
    
    # Start executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = env.create_rate(2)
    
    # Set up action space
    action_space = type('obj', (), {'shape': [1], 'high': np.array([1.0]), 'low': np.array([-1.0])})()
    
    # Initialize SAC agent
    agent = SAC(len(env.current_state), action_space, args)
    
    # Set up TensorBoard writer
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.env_name,
        args.policy,
        "autotune" if args.automatic_entropy_tuning else ""))
    
    # Initialize replay memory
    memory = ReplayMemory(args.replay_size, args.seed)
    
    # Training loop parameters
    total_numsteps = 0
    updates = 0
    max_episode_steps = 120
    
    # Lists to store metrics for learning curve
    episode_rewards = []
    episode_numbers = []
    eval_rewards = []
    eval_episode_numbers = []
    
    try:
        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Reset environment
            state = env.reset()
            
            while not done:
                print(f"Episode step: {episode_steps}, Total steps: {total_numsteps}/{args.num_steps}")
                
                # Select action based on exploration strategy
                if args.start_steps > total_numsteps:
                    action = 2 * np.random.rand(1) - 1  # Random action [-1, 1]
                else:
                    action = agent.select_action(state)  # Sample action from policy
                
                # Perform updates if enough samples in memory
                if len(memory) > args.batch_size:
                    for i in range(args.updates_per_step):
                        # Update agent parameters
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                            memory, args.batch_size, updates)
                        
                        # Log metrics
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temperature/alpha', alpha, updates)
                        updates += 1
                
                # Take a step in the environment
                next_state, reward, done = env.step(action, episode_steps, max_episode_steps)
                
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
                
                # Handle the "done" signal properly for terminal vs time-limit cases
                mask = 1 if episode_steps == max_episode_steps else float(not done)
                
                # Store transition in replay memory
                memory.push(state, action, reward, next_state, mask)
                
                # Update state
                state = next_state
                
                # Break if we've reached the step limit
                if total_numsteps > args.num_steps:
                    break

            # Log episode metrics
            writer.add_scalar('reward/train', episode_reward, i_episode)
            print("Episode: {}, Total steps: {}, Episode steps: {}, Reward: {}".format(
                i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
                            
            # Store training metrics for learning curve
            episode_rewards.append(episode_reward)
            episode_numbers.append(i_episode)
            
            # Clear CUDA cache for memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()            
            
            # Evaluation phase
            if i_episode % 10 == 0 and args.eval:
                avg_reward = 0.
                episodes = 10
                for i in range(episodes):
                    print(f"Evaluation episode {i}")
                    state = env.reset()
                    episode_reward = 0
                    eval_steps = 0
                    done = False
                    while not done:
                        action = agent.select_action(state, evaluate=True)
                        # Modified to match the main training loop
                        next_state, reward, done = env.step(action, eval_steps, max_episode_steps)
                        
                        episode_reward += reward
                        eval_steps += 1
                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= episodes
                
                writer.add_scalar('avg_reward/test', avg_reward, i_episode)
                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                print("----------------------------------------")

                # Store evaluation metrics for learning curve
                eval_rewards.append(avg_reward)
                eval_episode_numbers.append(i_episode)

                # Clear CUDA cache for memory management
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save model checkpoint periodically
            if i_episode % 20 == 0:
                agent.save_checkpoint(args.env_name, suffix=str(i_episode))
            
            # Break if we've completed all steps
            if total_numsteps > args.num_steps:
                break
                
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up
        writer.close()
        
        # Save learning curve if requested
        if args.save_curve and episode_numbers:
            save_learning_curve(episode_numbers, episode_rewards, 
                               eval_episode_numbers, eval_rewards, args.curve_name)
        
        print("Training finished")
if __name__ == "__main__":
    main()