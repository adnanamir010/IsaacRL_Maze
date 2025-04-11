#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import datetime
import os
import itertools
import gc
import gymnasium as gym
from tqdm import tqdm

from agents import SAC
from memory import ReplayMemory
from rl_utils import evaluate_policy, evaluate_policy_vec, save_learning_curve, collect_garbage
from environment import VectorizedDDEnv, make_vectorized_env
from torch.utils.tensorboard import SummaryWriter

def parse_arguments():
    """Parse command line arguments for the SAC algorithm"""
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="VectorizedDD",
                    help='Environment name (default: VectorizedDD)')
    parser.add_argument('--obstacle-shape', default="square",
                    help='Obstacle shape: circular | square (default: square)')
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
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automatically adjust α (default: True)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='Batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1_000_000, metavar='N',
                    help='Maximum number of steps (default: 1_000_000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='Hidden size (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='Model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=10, metavar='N',
                    help='Value target update interval (default: 10)')
    parser.add_argument('--replay_size', type=int, default=500_000, metavar='N',
                    help='Size of replay buffer (default: 500_000)')
    parser.add_argument('--cuda', action="store_true", default=True,
                    help='Run on CUDA (default: True)')
    parser.add_argument('--save-curve', action="store_true", default=True,
                    help='Save learning curve plot (default: True)')
    parser.add_argument('--curve-name', type=str, default='sac_vec_learning_curve',
                    help='Filename for learning curve plot (default: sac_vec_learning_curve)')
    parser.add_argument('--memory-efficient', action="store_true", default=True,
                    help='Enable memory efficiency optimizations (default: True)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                    help='Checkpoint save interval in episodes (default: 50)')
    parser.add_argument('--gc-interval', type=int, default=50,
                    help='Garbage collection interval in episodes (default: 50)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                    help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=20,
                    help='Evaluation interval in episodes (default: 20)')
    parser.add_argument('--log-interval', type=int, default=10,
                    help='Logging interval in episodes (default: 10)')
    parser.add_argument('--num-envs', type=int, default=4,
                    help='Number of parallel environments (default: 4)')
    parser.add_argument('--render', action="store_true",
                    help='Render visualization (only for training with num-envs=1) (default: False)')    
    parser.add_argument('--use-twin-critic', action="store_true", default=True,
                    help='Use twin critic (default: True)')
    parser.add_argument('--entropy-mode', choices=['none', 'fixed', 'adaptive'], default='adaptive',
                    help='Entropy mode: none, fixed, or adaptive (default: adaptive)')
    parser.add_argument('--experiment-name', type=str, default=None,
                    help='Name for this experiment (default: auto-generated)')

    return parser.parse_args()

def main():
    """Main training function with vectorized environments and progress bar"""
    # Parse arguments
    args = parse_arguments()
    
    # For backward compatibility, set entropy_mode based on automatic_entropy_tuning
    if hasattr(args, 'automatic_entropy_tuning'):
        if args.automatic_entropy_tuning:
            args.entropy_mode = 'adaptive'
        else:
            args.entropy_mode = 'fixed'
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        critic_type = "Twin" if args.use_twin_critic else "Single"
        entropy_type = args.entropy_mode.capitalize()
        args.experiment_name = f"{args.env_name}_{critic_type}Critic_{entropy_type}Entropy"
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Set GPU memory limits if using CUDA
        if args.memory_efficient:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    else:
        print("Using CPU")
    
    # Create vectorized environments
    print(f"Creating {args.num_envs} parallel environments for '{args.env_name}' with '{args.obstacle_shape}' obstacles")
    
    # Initialize render mode only if explicitly requested and only with a single environment
    render_mode = "human" if args.render and args.num_envs == 1 else None
    if args.render and args.num_envs > 1:
        print("Warning: Rendering is only supported for single environment training. Disabling rendering.")
    
    if args.env_name == "VectorizedDD":
        # Use our custom environment
        if args.num_envs == 1 and render_mode:
            # Single environment with rendering
            env_fn = lambda: VectorizedDDEnv(render_mode=render_mode, obstacle_shape=args.obstacle_shape)
            vec_env = gym.vector.SyncVectorEnv([env_fn])
        else:
            # Multiple environments or no rendering
            vec_env = make_vectorized_env(num_envs=args.num_envs, seed=args.seed, obstacle_shape=args.obstacle_shape)
        
        # Create a single environment for evaluation
        eval_env = make_vectorized_env(num_envs=1, seed=args.seed + 100, obstacle_shape=args.obstacle_shape)
    else:
        # Use standard Gym environment
        vec_env = gym.vector.make(args.env_name, num_envs=args.num_envs, asynchronous=False)
        eval_env = gym.vector.make(args.env_name, num_envs=1, asynchronous=False)
    
    # Get state and action dimensions
    state_dim = vec_env.single_observation_space.shape[0]
    
    # Handle different types of action spaces
    if isinstance(vec_env.single_action_space, gym.spaces.Box):
        action_dim = vec_env.single_action_space.shape[0]
        print(f"Continuous action space detected with {action_dim} dimensions")
    elif isinstance(vec_env.single_action_space, gym.spaces.Discrete):
        action_dim = 1  # Discrete action is represented as a single integer
        print(f"Discrete action space detected with {vec_env.single_action_space.n} possible actions")
    else:
        raise ValueError(f"Unsupported action space type: {type(vec_env.single_action_space)}")
        
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Set up action space
    action_space = vec_env.single_action_space
    
    # Log configuration details
    print(f"\nSAC Configuration:")
    print(f"- Twin Critic: {args.use_twin_critic}")
    print(f"- Entropy Mode: {args.entropy_mode}")
    if args.entropy_mode == 'fixed':
        print(f"- Alpha Value: {args.alpha}")
    print()
    
    # Initialize SAC agent
    agent = SAC(state_dim, action_space, args)
    
    # Set up TensorBoard writer with reduced flush frequency to minimize I/O
    log_dir = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.experiment_name}'
    writer = SummaryWriter(log_dir, max_queue=1000, flush_secs=120)
    
    # Create replay memory with state and action dimensions
    memory = ReplayMemory(args.replay_size, state_dim, action_dim, args.seed)
    
    # Training loop parameters
    total_numsteps = 0
    updates = 0
    
    # To keep track of episodes in vectorized environment
    episode_rewards = [0] * args.num_envs
    episode_steps = [0] * args.num_envs
    episode_count = 0
    
    # Lists to store metrics for learning curve
    all_episode_rewards = []
    all_episode_numbers = []
    all_eval_rewards = []
    all_eval_episode_numbers = []
    
    # Initialize progress bar for total training
    progress_bar = tqdm(total=args.num_steps, desc="Training Progress", unit="steps")
    last_update = 0
    
    # Reset environments to get initial states
    states, _ = vec_env.reset(seed=args.seed)
    
    try:
        while total_numsteps < args.num_steps:
            # Run garbage collection periodically
            if total_numsteps % (args.gc_interval * args.num_envs) == 0:
                collect_garbage()
            
            # Select actions
            if total_numsteps < args.start_steps:
                # Sample random actions for initial exploration
                if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                    # For discrete action space
                    actions = np.array([vec_env.single_action_space.sample() for _ in range(args.num_envs)])
                else:
                    # For continuous action space
                    actions = np.array([vec_env.single_action_space.sample() for _ in range(args.num_envs)])
            else:
                # Sample actions from policy
                actions = agent.select_actions_vec(states, evaluate=False)

            if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                # Convert from array form [n] to scalar form n
                if isinstance(actions, np.ndarray):
                    if actions.ndim == 2 and actions.shape[1] == 1:
                        # If shape is (n_envs, 1)
                        actions = actions.flatten().astype(np.int32)
                    elif actions.ndim == 1:
                        # If shape is (n_envs,)
                        actions = actions.astype(np.int32)
                
            # Take steps in environments
            next_states, rewards, terminations, truncations, infos = vec_env.step(actions)
            
            # Combine terminations and truncations
            dones = np.logical_or(terminations, truncations)
            
            # Collect transitions in replay buffer
            for i in range(args.num_envs):
                # Handle the "done" signal properly for terminal vs time-limit cases
                # Most gym environments set done=True for both episode completion and time limits
                # Here we assume all dones are true terminal states for simplicity
                terminal_done = dones[i]
                mask = 0.0 if terminal_done else 1.0  # Terminal state = 0, non-terminal = 1
                
                # Store transition in replay memory
                if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                    # Convert discrete action to one-hot vector or keep as scalar
                    discrete_action = actions[i]
                    memory.push(states[i], np.array([discrete_action]), rewards[i], next_states[i], mask)
                else:
                    # Continuous action
                    memory.push(states[i], actions[i], rewards[i], next_states[i], mask)
                
                # Update episode statistics
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                
                # Handle episode termination
                if dones[i]:
                    # Log episode stats
                    episode_count += 1
                    writer.add_scalar('reward/train', episode_rewards[i], episode_count)
                    
                    # Store training metrics for learning curve
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_numbers.append(episode_count)
                    
                    # Print progress
                    tqdm.write(f"Episode {episode_count}: Reward={episode_rewards[i]:.2f}, Steps={episode_steps[i]}")
                    
                    # Reset episode stats
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    
                    # Run evaluation periodically
                    if episode_count % args.eval_interval == 0 and args.eval:
                        # Set evaluation message
                        progress_bar.set_description("Evaluating...")
                        
                        # Run evaluation
                        eval_mean_reward, eval_std_reward = evaluate_policy_vec(
                            agent, eval_env, max_eval_episodes=args.eval_episodes)
                        
                        # Log evaluation metrics
                        writer.add_scalar('reward/eval', eval_mean_reward, episode_count)
                        all_eval_rewards.append(eval_mean_reward)
                        all_eval_episode_numbers.append(episode_count)
                        
                        # Print evaluation results
                        progress_bar.write(f"Evaluation: {args.eval_episodes} episodes, " 
                                          f"Mean Reward: {eval_mean_reward:.2f}, "
                                          f"Std: {eval_std_reward:.2f}")
                        
                        # Reset progress bar description
                        progress_bar.set_description("Training Progress")
                        
                        # Run garbage collection after evaluation
                        collect_garbage()
                    
                    # Save model checkpoint periodically
                    if episode_count % args.checkpoint_interval == 0:
                        agent.save_checkpoint(args.env_name, suffix=f"{args.experiment_name}_{episode_count}")
                        
                        # Also save replay buffer occasionally for potential resume
                        if episode_count % (args.checkpoint_interval * 5) == 0:
                            memory.save_buffer(args.env_name, suffix=f"{args.experiment_name}_{episode_count}")
            
            # Perform updates if enough samples in memory
            if len(memory) > args.batch_size:
                # Number of updates per step
                for _ in range(args.updates_per_step * args.num_envs):
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
            
            # Update states
            states = next_states
            total_numsteps += args.num_envs
            
            # Update progress bar
            progress_bar.update(args.num_envs)
            progress_bar.set_postfix({
                'episodes': episode_count,
                'steps': total_numsteps
            })
            
    except KeyboardInterrupt:
        progress_bar.write("\nTraining interrupted by user")
    finally:
        # Close progress bar
        progress_bar.close()
        
        # Final save
        if 'episode_count' in locals() and episode_count > 0:
            print(f"Saving final checkpoint at episode {episode_count}")
            agent.save_checkpoint(args.env_name, suffix=f"{args.experiment_name}_final_{episode_count}")
            memory.save_buffer(args.env_name, suffix=f"{args.experiment_name}_final_{episode_count}")
        
        # Clean up
        writer.close()
        vec_env.close()
        eval_env.close()
        
        # Save learning curve if requested
        if args.save_curve and all_episode_numbers:
            # Convert lists to NumPy arrays for efficient processing
            episode_numbers_np = np.array(all_episode_numbers)
            episode_rewards_np = np.array(all_episode_rewards)
            
            eval_episode_numbers_np = np.array(all_eval_episode_numbers) if all_eval_episode_numbers else None
            eval_rewards_np = np.array(all_eval_rewards) if all_eval_rewards else None
            
            # Include configuration in the curve name
            curve_name = f"{args.curve_name}_{args.experiment_name}"
            algorithm_name = f"{critic_type} Critic SAC with {entropy_type} Entropy"
            save_learning_curve(episode_numbers_np, episode_rewards_np, 
                               eval_episode_numbers_np, eval_rewards_np, curve_name, algorithm_name)
        
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