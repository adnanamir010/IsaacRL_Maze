#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

# Import your modules
from agents import SAC
from memory import ReplayMemory
from environment import VectorizedDDEnv, make_vectorized_env
from rl_utils import evaluate_policy_vec, collect_garbage

def parse_arguments():
    """Parse command line arguments for the SAC experiments"""
    parser = argparse.ArgumentParser(description='Run SAC experiments with different configurations')
    
    # Environment parameters
    parser.add_argument('--env-name', default="VectorizedDD", help='Environment name (default: VectorizedDD)')
    parser.add_argument('--seed', type=int, default=123456, help='Random seed (default: 123456)')
    parser.add_argument('--num-envs', type=int, default=2, help='Number of parallel environments (default: 2)')
    parser.add_argument('--obstacle-shape', default="square", help='Obstacle shape: circular | square (default: square)')
    
    # Training parameters
    parser.add_argument('--num-steps', type=int, default=500_000, help='Maximum number of steps (default: 1_000_000)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--start-steps', type=int, default=5000, help='Steps sampling random actions (default: 5000)')
    parser.add_argument('--updates-per-step', type=int, default=1, help='Updates per step (default: 1)')
    
    # Evaluation parameters
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluation interval in episodes (default: 50)')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes per evaluation (default: 5)')
    
    # SAC parameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, help='Target smoothing coefficient (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Fixed entropy coefficient (default: 0.2)')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size (default: 128)')
    parser.add_argument('--target-update-interval', type=int, default=10, help='Target update interval (default: 10)')
    parser.add_argument('--replay-size', type=int, default=500_000, help='Replay buffer size (default: 500_000)')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    
    # Configuration parameters
    parser.add_argument('--run-all', action='store_true', help='Run all 6 experiments')
    parser.add_argument('--use-twin-critic', action='store_true', help='Use twin critic (default: False)')
    parser.add_argument('--entropy-mode', choices=['none', 'fixed', 'adaptive'], default='adaptive',
                       help='Entropy mode: none, fixed, or adaptive (default: adaptive)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available (default: False)')
    
    # Output parameters
    parser.add_argument('--output-dir', default='experiment_results', help='Directory for output files')
    
    return parser.parse_args()

def train_sac(args, variant_name):
    """
    Train a SAC agent with the given configuration.
    
    Args:
        args: Command line arguments
        variant_name: Name of the variant for logging
        
    Returns:
        tuple: (episode_rewards, eval_rewards, episode_numbers, eval_episode_numbers)
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environments
    vec_env = make_vectorized_env(num_envs=args.num_envs, seed=args.seed, obstacle_shape=args.obstacle_shape)
    eval_env = make_vectorized_env(num_envs=1, seed=args.seed + 100, obstacle_shape=args.obstacle_shape)
    
    # Get dimensions
    state_dim = vec_env.single_observation_space.shape[0]
    action_space = vec_env.single_action_space
    
    # Create agent
    agent = SAC(state_dim, action_space, args)
    
    # Create replay memory
    if isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
    else:
        action_dim = 1
    memory = ReplayMemory(args.replay_size, state_dim, action_dim, args.seed)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(f"{args.output_dir}/runs/{variant_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    # Training tracking variables
    total_numsteps = 0
    updates = 0
    episode_rewards = [0] * args.num_envs
    episode_steps = [0] * args.num_envs
    episode_count = 0
    
    # Lists to store metrics
    all_episode_rewards = []
    all_episode_numbers = []
    all_eval_rewards = []
    all_eval_episode_numbers = []
    
    # Reset environments
    states, _ = vec_env.reset(seed=args.seed)
    
    # Create progress bar
    progress_bar = tqdm(total=args.num_steps, desc=f"Training {variant_name}", unit="steps")
    
    # Training loop
    try:
        while total_numsteps < args.num_steps:
            # Select actions
            if total_numsteps < args.start_steps:
                # Random actions for initial exploration
                if isinstance(action_space, gym.spaces.Discrete):
                    actions = np.array([action_space.sample() for _ in range(args.num_envs)])
                else:
                    actions = np.array([action_space.sample() for _ in range(args.num_envs)])
            else:
                # Sample from policy
                actions = agent.select_actions_vec(states, evaluate=False)
            
            # Format actions for environment
            if isinstance(action_space, gym.spaces.Discrete):
                if isinstance(actions, np.ndarray):
                    if actions.ndim == 2 and actions.shape[1] == 1:
                        actions_env = actions.flatten().astype(np.int32)
                    else:
                        actions_env = actions.astype(np.int32)
                else:
                    actions_env = actions
            else:
                actions_env = actions
            
            # Take environment step
            next_states, rewards, terminations, truncations, infos = vec_env.step(actions_env)
            dones = np.logical_or(terminations, truncations)
            
            # Store transitions in memory
            for i in range(args.num_envs):
                mask = 0.0 if dones[i] else 1.0
                
                if isinstance(action_space, gym.spaces.Discrete):
                    discrete_action = actions[i]
                    if hasattr(discrete_action, '__len__'):
                        discrete_action = discrete_action[0]
                    memory.push(states[i], np.array([discrete_action]), rewards[i], next_states[i], mask)
                else:
                    memory.push(states[i], actions[i], rewards[i], next_states[i], mask)
                
                # Update episode tracking
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                
                # Handle episode termination
                if dones[i]:
                    episode_count += 1
                    
                    # Store training metrics
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_numbers.append(episode_count)
                    
                    # Log to TensorBoard
                    writer.add_scalar('train/episode_reward', episode_rewards[i], episode_count)
                    writer.add_scalar('train/episode_steps', episode_steps[i], episode_count)
                    
                    # Print progress
                    progress_bar.write(f"Episode {episode_count}: Reward={episode_rewards[i]:.2f}, Steps={episode_steps[i]}")
                    
                    # Reset episode stats
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
                    
                    # Run evaluation periodically
                    if episode_count % args.eval_interval == 0:
                        eval_mean_reward, eval_std_reward = evaluate_policy_vec(
                            agent, eval_env, max_eval_episodes=args.eval_episodes)
                        
                        # Store evaluation metrics
                        all_eval_rewards.append(eval_mean_reward)
                        all_eval_episode_numbers.append(episode_count)
                        
                        # Log to TensorBoard
                        writer.add_scalar('eval/mean_reward', eval_mean_reward, episode_count)
                        writer.add_scalar('eval/std_reward', eval_std_reward, episode_count)
                        
                        progress_bar.write(f"Evaluation: {args.eval_episodes} episodes, "
                                         f"Mean Reward: {eval_mean_reward:.2f}")
            
            # Update agent if enough samples
            if len(memory) > args.batch_size:
                # Number of updates per step
                for _ in range(args.updates_per_step * args.num_envs):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates)
                    
                    # Log training metrics
                    if updates % 100 == 0:
                        writer.add_scalar('train/critic_1_loss', critic_1_loss, updates)
                        writer.add_scalar('train/critic_2_loss', critic_2_loss, updates)
                        writer.add_scalar('train/policy_loss', policy_loss, updates)
                        writer.add_scalar('train/entropy_loss', ent_loss, updates)
                        writer.add_scalar('train/alpha', alpha, updates)
                    
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
        # Close progress bar and environments
        progress_bar.close()
        vec_env.close()
        eval_env.close()
        writer.close()
        
        # Save the agent
        agent.save_checkpoint(args.env_name, suffix=f"{variant_name}")
        
        return all_episode_rewards, all_eval_rewards, all_episode_numbers, all_eval_episode_numbers

def plot_comparison(results, filename):
    """
    Plot comparison of different SAC variants.
    
    Args:
        results: Dictionary mapping variant names to (episode_rewards, eval_rewards, episode_numbers, eval_episode_numbers)
        filename: Output filename
    """
    plt.figure(figsize=(15, 10))
    
    # Colors for different variants
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    # Plot training rewards
    plt.subplot(2, 1, 1)
    
    for i, (variant_name, (ep_rewards, _, ep_nums, _)) in enumerate(results.items()):
        # Skip if no data
        if not ep_rewards:
            continue
        
        # Convert to numpy arrays
        ep_nums = np.array(ep_nums)
        ep_rewards = np.array(ep_rewards)
        
        # Plot raw data with alpha
        plt.plot(ep_nums, ep_rewards, color=colors[i % len(colors)], alpha=0.2)
        
        # Plot smoothed data
        window_size = min(len(ep_rewards) // 5, 10) if len(ep_rewards) > 10 else 1
        if window_size > 1:
            smoothed_rewards = np.zeros_like(ep_rewards)
            for j in range(len(ep_rewards)):
                start_idx = max(0, j - window_size)
                end_idx = min(len(ep_rewards), j + window_size + 1)
                smoothed_rewards[j] = np.mean(ep_rewards[start_idx:end_idx])
                
            # Plot smoothed line
            plt.plot(ep_nums, smoothed_rewards, color=colors[i % len(colors)], 
                    linewidth=2, label=variant_name)
        else:
            # If we don't have enough data for smoothing
            plt.plot(ep_nums, ep_rewards, color=colors[i % len(colors)], 
                    linewidth=2, label=variant_name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Training Episode Reward')
    plt.title('Training Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot evaluation rewards
    plt.subplot(2, 1, 2)
    
    for i, (variant_name, (_, eval_rewards, _, eval_ep_nums)) in enumerate(results.items()):
        if not eval_rewards:  # Skip if no data
            continue
            
        # Convert to numpy arrays
        eval_ep_nums = np.array(eval_ep_nums)
        eval_rewards = np.array(eval_rewards)
        
        # Plot evaluation data
        plt.plot(eval_ep_nums, eval_rewards, color=colors[i % len(colors)], 
                marker='o', linewidth=2, label=variant_name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Evaluation Mean Reward')
    plt.title('Evaluation Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"Comparison plot saved to {filename}")
    plt.close()

def run_experiments(args):
    """
    Run SAC experiments based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Dictionary to store results
    results = {}
    
    if args.run_all:
        # Run all 6 combinations
        variants = [
            (True, "adaptive", "Twin Critic + Adaptive Entropy"),
            (True, "fixed", "Twin Critic + Fixed Entropy"),
            (True, "none", "Twin Critic + No Entropy"),
            (False, "adaptive", "Single Critic + Adaptive Entropy"),
            (False, "fixed", "Single Critic + Fixed Entropy"),
            (False, "none", "Single Critic + No Entropy")
        ]
        
        for use_twin_critic, entropy_mode, variant_name in variants:
            # Update args with variant config
            args.use_twin_critic = use_twin_critic
            args.entropy_mode = entropy_mode
            
            print(f"\n{'='*50}")
            print(f"Training variant: {variant_name}")
            print(f"{'='*50}\n")
            
            # Train the variant
            results[variant_name] = train_sac(args, variant_name)
            
            # Force garbage collection between runs
            collect_garbage()
            
        # Plot comparison of all variants
        plot_comparison(results, f"{args.output_dir}/sac_variants_comparison.png")
    else:
        # Run a single experiment with specified config
        critic_type = "Twin" if args.use_twin_critic else "Single"
        entropy_type = args.entropy_mode.capitalize()
        variant_name = f"{critic_type} Critic + {entropy_type} Entropy"
        
        results[variant_name] = train_sac(args, variant_name)

def main():
    """Main function to run SAC experiments"""
    args = parse_arguments()
    run_experiments(args)

if __name__ == "__main__":
    main()