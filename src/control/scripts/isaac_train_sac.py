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
import gc
from tqdm import tqdm, trange

# Initialize Isaac Sim Application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Import Isaac Sim modules
import omni
from omni.isaac.core import World

# Import required modules
from isaac_agents import SAC
from memory import ReplayMemory
from isaac_environment import DDEnv
from isaac_utils import ModelStateSubscriber, LidarSubscriber, CollisionDetector, save_learning_curve, evaluate_policy
from torch.utils.tensorboard import SummaryWriter
import global_vars

# Configure PyTorch memory management
torch.backends.cudnn.benchmark = True  # Optimize CUDNN
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better performance
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for CUDNN

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

# Initialize global variables - using NumPy arrays with specific data types
global_vars.body_pose = np.zeros(3, dtype=np.float32)  # x, y, theta
global_vars.lidar_data = np.zeros(20, dtype=np.float32)
global_vars.clash_sum = 0

# Image settings for visualization and collision detection
image_size = 720
pixel_to_meter = 0.1  # m/pix
wheelbase = 8  # length of robot
stage_width = image_size * pixel_to_meter
stage_height = image_size * pixel_to_meter

# Set numpy print format for better readability
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

# Memory management utilities
def clear_cuda_cache():
    """Clear CUDA cache to free up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def collect_garbage():
    """Force garbage collection"""
    gc.collect()
    clear_cuda_cache()

def parse_arguments():
    """Parse command line arguments for the SAC algorithm"""
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
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
    parser.add_argument('--num_steps', type=int, default=500_000, metavar='N',
                    help='Maximum number of steps (default: 1000001)')
    parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                    help='Hidden size (default: 64)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='Model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=4000, metavar='N',
                    help='Steps sampling random actions (default: 4000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update interval (default: 1)')
    parser.add_argument('--replay_size', type=int, default=100_000, metavar='N',
                    help='Size of replay buffer (default: 100_000)')
    parser.add_argument('--cuda', action="store_true",
                    help='Run on CUDA (default: False)')
    parser.add_argument('--save-curve', action="store_true",
                    help='Save learning curve plot (default: False)')
    parser.add_argument('--curve-name', type=str, default='learning_curve',
                    help='Filename for learning curve plot (default: learning_curve)')
    parser.add_argument('--memory-efficient', action="store_true", default=True,
                    help='Enable memory efficiency optimizations (default: True)')
    parser.add_argument('--checkpoint-interval', type=int, default=20,
                    help='Checkpoint save interval in episodes (default: 20)')
    parser.add_argument('--gc-interval', type=int, default=50,
                    help='Garbage collection interval in episodes (default: 50)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                    help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=50,
                    help='Evaluation interval in episodes (default: 50)')
    parser.add_argument('--log-interval', type=int, default=5,
                    help='Logging interval in episodes (default: 5)')
    parser.add_argument('--num-episodes', type=int, default=1000,
                help='Maximum number of episodes (default: 1000)')
    return parser.parse_args()

def main():
    """Main training function with memory optimization and progress bar"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize ROS2
    rclpy.init(args=None)
    
    # Create environment and subscribers
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
    
    # Configure device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Set GPU memory limits if using CUDA
        if args.memory_efficient:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    else:
        print("Using CPU")
    
    # Set up TensorBoard writer with reduced flush frequency to minimize I/O
    log_dir = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.env_name}_{args.policy}'
    writer = SummaryWriter(log_dir, max_queue=1000, flush_secs=120)
    
    # Create replay memory with state and action dimensions
    state_dim = len(env.current_state)
    action_dim = action_space.shape[0]
    memory = ReplayMemory(args.replay_size, state_dim, action_dim, args.seed)
    
    # Training loop parameters
    total_numsteps = 0
    updates = 0
    max_episode_steps = 220
    
    # Lists to store metrics for learning curve (using NumPy arrays for efficiency)
    episode_rewards = []
    episode_numbers = []
    eval_rewards = []
    eval_episode_numbers = []
    
    # Initialize progress bar for total training
    from tqdm import tqdm
    progress_bar = tqdm(total=args.num_steps, desc="Training Progress", unit="steps")
    last_update = 0
    
    try:
        for i_episode in itertools.count(1):
            # Run garbage collection periodically to free memory
            if i_episode % args.gc_interval == 0:
                collect_garbage()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Reset environment
            state = env.reset()
            
            # Episode loop
            while not done:
                # Select action based on exploration strategy
                if args.start_steps > total_numsteps:
                    # Random action [-1, 1] - use numpy for efficiency
                    action = np.random.uniform(-1, 1, (1,)).astype(np.float32)
                else:
                    # Sample action from policy - with no_grad() to save memory
                    with torch.no_grad():
                        action = agent.select_action(state)
                
                # Perform updates if enough samples in memory - with memory optimizations
                if len(memory) > args.batch_size:
                    # Limit updates to reduce memory pressure during exploration
                    max_updates = min(args.updates_per_step, 5 if total_numsteps < 50000 else 10)
                    
                    for i in range(max_updates):
                        # Update agent parameters - wrap in no_grad for critic evaluation
                        with torch.set_grad_enabled(True):
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                                memory, args.batch_size, updates)
                        
                        # Log metrics only periodically to reduce overhead
                        if updates % 100 == 0:
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
                
                # Update progress bar
                if total_numsteps > last_update:
                    progress_bar.update(total_numsteps - last_update)
                    progress_bar.set_postfix({
                        'episode': i_episode,
                        'ep_steps': episode_steps,
                        'reward': f"{episode_reward:.2f}"
                    })
                    last_update = total_numsteps
                
                # Handle the "done" signal properly for terminal vs time-limit cases
                mask = 1 if episode_steps == max_episode_steps else float(not done)
                
                # Store transition in replay memory
                memory.push(state, action, reward, next_state, mask)
                
                # Update state
                state = next_state
                
                # Break if we've reached the step limit
                if total_numsteps > args.num_steps:
                    break
            
            # Log episode metrics - only log periodically to reduce I/O
            if i_episode % args.log_interval == 0:
                writer.add_scalar('reward/train', episode_reward, i_episode)
                
            # Store training metrics for learning curve (convert to NumPy array at the end)
            episode_rewards.append(episode_reward)
            episode_numbers.append(i_episode)
            
            # Evaluation phase - only run periodically to reduce overhead
            if i_episode % args.eval_interval == 0 and args.eval:
                # Temporarily pause the progress bar to display evaluation message
                progress_bar.set_description("Evaluating...")
                
                # Run evaluation episodes
                eval_rewards_batch = []
                for eval_ep in range(args.eval_episodes):
                    eval_reward = evaluate_policy(agent, env, max_episode_steps, eval_ep)
                    eval_rewards_batch.append(eval_reward)
                
                # Calculate average reward
                avg_reward = np.mean(eval_rewards_batch)
                
                writer.add_scalar('avg_reward/test', avg_reward, i_episode)
                progress_bar.write(f"Evaluation: {args.eval_episodes} episodes, Avg. Reward: {avg_reward:.2f}")
                
                # Store evaluation metrics for learning curve
                eval_rewards.append(avg_reward)
                eval_episode_numbers.append(i_episode)
                
                # Run garbage collection after evaluation
                collect_garbage()
                
                # Reset progress bar description
                progress_bar.set_description("Training Progress")
            
            # Save model checkpoint periodically
            if i_episode % args.checkpoint_interval == 0:
                agent.save_checkpoint(args.env_name, suffix=str(i_episode))
                
                # Also save replay buffer occasionally for potential resume
                if i_episode % (args.checkpoint_interval * 5) == 0:
                    memory.save_buffer(args.env_name, suffix=str(i_episode))
            
            # Break if we've completed all steps
            if total_numsteps > args.num_steps:
                break
                
    except KeyboardInterrupt:
        progress_bar.write("\nTraining interrupted by user")
    finally:
        # Close progress bar
        progress_bar.close()
        
        # Final save
        final_episode = i_episode if 'i_episode' in locals() else 0
        if final_episode > 0:
            print(f"Saving final checkpoint at episode {final_episode}")
            agent.save_checkpoint(args.env_name, suffix=f"final_{final_episode}")
            memory.save_buffer(args.env_name, suffix=f"final_{final_episode}")
        
        # Clean up
        writer.close()
        
        # Save learning curve if requested
        if args.save_curve and episode_numbers:
            # Convert lists to NumPy arrays for efficient processing
            episode_numbers_np = np.array(episode_numbers)
            episode_rewards_np = np.array(episode_rewards)
            
            eval_episode_numbers_np = np.array(eval_episode_numbers) if eval_episode_numbers else None
            eval_rewards_np = np.array(eval_rewards) if eval_rewards else None
            
            save_learning_curve(episode_numbers_np, episode_rewards_np, 
                               eval_episode_numbers_np, eval_rewards_np, args.curve_name)
        
        # Final cleanup
        collect_garbage()
        print("Training finished")

if __name__ == "__main__":
    # Set lower memory usage for numpy operations
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    
    # Set up environment variables to limit CPU utilization if needed
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
    
    # Set CUDA options to optimize memory usage
    if torch.cuda.is_available():
        # Try to ensure PyTorch releases memory after operations
        torch.cuda.empty_cache()
        
    # Run main function
    main()