#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import datetime
import os
import gc
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from agents import PPOCLIP, PPOKL
from memory import RolloutStorage
from rl_utils import evaluate_policy_vec, collect_garbage
from environment import VectorizedDDEnv, make_vectorized_env
from torch.utils.tensorboard import SummaryWriter

def parse_arguments():
    """Parse command line arguments for the PPO algorithm variants"""
    parser = argparse.ArgumentParser(description='PyTorch PPO Algorithm Args')
    
    # Algorithm choice
    parser.add_argument('--algorithm', type=str, default="PPOCLIP", choices=["PPOCLIP", "PPOKL"],
                    help='PPO algorithm variant: PPOCLIP | PPOKL (default: PPOCLIP)')
    
    # Environment parameters
    parser.add_argument('--env-name', default="VectorizedDD",
                    help='Environment name (default: VectorizedDD)')
    parser.add_argument('--obstacle-shape', default="square",
                help='Obstacle shape: circular | square (default: square)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--num-envs', type=int, default=16,
                    help='Number of parallel environments (default: 16)')
    
    # General PPO parameters (shared)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='GAE parameter (default: 0.95)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                    help='Hidden layer size (default: 256)')
    parser.add_argument('--batch-size', type=int, default=8192, metavar='N',
                    help='Batch size for PPO updates (default: 8192)')
    parser.add_argument('--num-steps', type=int, default=5_000_000, metavar='N',
                    help='Maximum number of steps (default: 5_000_000)')
    parser.add_argument('--update-interval', type=int, default=2048, metavar='N',
                    help='Steps between PPO updates (default: 2048)')
    parser.add_argument('--num-mini-batch', type=int, default=4, metavar='G',
                    help='Number of PPO mini-batches (default: 4)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, metavar='G',
                    help='Max gradient norm (default: 0.5)')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                    help='Model updates per simulator step (default: 1)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, metavar='G',
                help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--use-clipped-value-loss', action='store_true', default=True,
                help='Use clipped value loss (default: True)')

    # PPOCLIP specific parameters
    parser.add_argument('--clip-param', type=float, default=0.1, metavar='G',
                    help='PPO clip parameter (default: 0.1)')
    parser.add_argument('--policy-lr', type=float, default=2.5e-4, metavar='G',
                    help='Policy learning rate (default: 2.5e-4)')
    parser.add_argument('--value-lr', type=float, default=5e-4, metavar='G',
                    help='Value function learning rate (default: 5e-4)')  
    parser.add_argument('--ppo-epoch', type=int, default=4, metavar='G',
                    help='Number of PPO epochs (default: 4)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, metavar='G',
                    help='Entropy coefficient (default: 0.01)')
    
    # PPOKL specific parameters
    parser.add_argument('--kl-target', type=float, default=0.005,
                    help='Target KL divergence (default: 0.005)')
    parser.add_argument('--kl-beta', type=float, default=1.0,
                    help='Initial KL coefficient (default: 1.0)')
    parser.add_argument('--kl-adaptive', action='store_true', default=True,
                    help='Use adaptive KL coefficient (default: True)')
    parser.add_argument('--kl-cutoff-factor', type=float, default=4.0,
                    help='KL cutoff factor (default: 4.0)')
    parser.add_argument('--kl-cutoff-coef', type=float, default=2000.0,
                    help='KL cutoff coefficient (default: 2000.0)')
    parser.add_argument('--min-kl-coef', type=float, default=0.02,
                    help='Minimum KL coefficient (default: 0.02)')
    parser.add_argument('--max-kl-coef', type=float, default=5.0,
                help='Minimum KL coefficient (default: 5.0)')
    
    # System and utilities
    parser.add_argument('--cuda', action="store_true", default=True,
                    help='Run on CUDA (default: True)')
    parser.add_argument('--target-update-interval', type=int, default=10, metavar='N',
                    help='Value target update interval (default: 10)')
    parser.add_argument('--save-curve', action="store_true", default=True,
                    help='Save learning curve plot (default: True)')
    parser.add_argument('--curve-name', type=str, default=None,
                    help='Filename for learning curve plot (auto generated if None)')
    parser.add_argument('--memory-efficient', action="store_true", default=True,
                    help='Enable memory efficiency optimizations (default: True)')
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                    help='Checkpoint save interval in updates (default: 50)')
    parser.add_argument('--gc-interval', type=int, default=10,
                    help='Garbage collection interval in updates (default: 10)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                    help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=5,
                    help='Evaluation interval in updates (default: 5)')
    parser.add_argument('--log-interval', type=int, default=1,
                    help='Logging interval in updates (default: 1)')
    parser.add_argument('--render', action="store_true",
                    help='Render visualization (only for training with num-envs=1) (default: False)')
    parser.add_argument('--experiment-name', type=str, default=None,
                    help='Name for this experiment (default: auto-generated)')
    parser.add_argument('--lr-annealing', action="store_true", default=True,
                    help='Use learning rate annealing (default: True)')
    parser.add_argument('--normalize-advantages', action="store_true", default=True,
                    help='Normalize advantages (default: True)')
    parser.add_argument('--verbose', action="store_true", default=False,
                    help='Print detailed logs (default: False)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Adjust parameters based on algorithm choice
    if args.algorithm == "PPOKL":
        # Override with PPOKL-specific optimal parameters
        args.policy_lr = 1e-4
        args.value_lr = 3e-4
        args.ppo_epoch = 5
        args.entropy_coef = 0.003
        args.gamma = 0.995
    
    return args

def save_enhanced_learning_curve(episodes, rewards, eval_episodes=None, eval_rewards=None, 
                                 value_losses=None, policy_losses=None, entropies=None, 
                                 clip_fractions=None, kl_divergences=None, 
                                 filename='learning_curve', algorithm='PPO'):
    """
    Save an enhanced plot of the learning curve with multiple metrics and statistics.
    
    Args:
        episodes: List of episode numbers for training
        rewards: List of training episode rewards
        eval_episodes: List of episode numbers for evaluation
        eval_rewards: List of evaluation rewards
        value_losses: List of value losses
        policy_losses: List of policy losses
        entropies: List of entropies
        clip_fractions: List of clip fractions (for PPO-CLIP)
        kl_divergences: List of KL divergences (for PPO-KL)
        filename: Filename for the saved plot
        algorithm: Algorithm name for the title
    """
    # Convert to numpy arrays
    episodes = np.array(episodes)
    rewards = np.array(rewards)
    
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 12), dpi=100)
    
    # Create a grid of subplots
    grid_size = 3
    if kl_divergences is not None:
        grid_size = 4  # Add extra row for KL divergence
    
    # 1. Plot training rewards with statistics
    plt.subplot(grid_size, 2, 1)
    plt.plot(episodes, rewards, 'b-', alpha=0.2, label='Episode Rewards')
    
    # Add smoothed version of training rewards with confidence bands
    window_size = min(len(rewards) // 5, 100)
    if window_size > 1:
        smoothed_rewards = np.zeros_like(rewards)
        confidence_upper = np.zeros_like(rewards)
        confidence_lower = np.zeros_like(rewards)
        
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(rewards), i + window_size + 1)
            window_rewards = rewards[start_idx:end_idx]
            
            mean_reward = np.mean(window_rewards)
            smoothed_rewards[i] = mean_reward
            
            if len(window_rewards) > 1:
                std_dev = np.std(window_rewards)
                confidence_upper[i] = mean_reward + 1.96 * std_dev / np.sqrt(len(window_rewards))
                confidence_lower[i] = mean_reward - 1.96 * std_dev / np.sqrt(len(window_rewards))
            else:
                confidence_upper[i] = mean_reward
                confidence_lower[i] = mean_reward
        
        plt.plot(episodes, smoothed_rewards, 'b-', linewidth=2, 
                 label=f'Smoothed (window={window_size*2+1})')
        plt.fill_between(episodes, confidence_lower, confidence_upper, 
                         color='b', alpha=0.2, label='95% CI')
    
    # Plot best and worst episodes
    max_idx = np.argmax(rewards)
    min_idx = np.argmin(rewards)
    plt.plot(episodes[max_idx], rewards[max_idx], 'g*', markersize=10, 
             label=f'Best: {rewards[max_idx]:.1f}')
    plt.plot(episodes[min_idx], rewards[min_idx], 'r*', markersize=10, 
             label=f'Worst: {rewards[min_idx]:.1f}')
    
    # Add mean and recent averages
    mean_reward = np.mean(rewards)
    if len(rewards) >= 100:
        recent_mean = np.mean(rewards[-100:])
        plt.axhline(y=recent_mean, color='m', linestyle='--', alpha=0.8,
                    label=f'Recent Mean: {recent_mean:.1f}')
    
    plt.axhline(y=mean_reward, color='k', linestyle='--', alpha=0.5,
                label=f'Overall Mean: {mean_reward:.1f}')
    
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.title('Training Rewards')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    # 2. Plot evaluation rewards if available
    plt.subplot(grid_size, 2, 2)
    if eval_episodes is not None and eval_rewards is not None:
        eval_episodes = np.array(eval_episodes)
        eval_rewards = np.array(eval_rewards)
        
        plt.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, marker='o', label='Evaluation')
        
        # If we have multiple evaluation points, add a trend line
        if len(eval_rewards) > 3:
            z = np.polyfit(eval_episodes, eval_rewards, 1)
            p = np.poly1d(z)
            plt.plot(eval_episodes, p(eval_episodes), 'r--', 
                     label=f'Trend: {z[0]:.2e}x + {z[1]:.1f}')
            
            # Add prediction line for future performance
            if len(eval_episodes) > 0:
                last_episode = eval_episodes[-1]
                future_episodes = np.linspace(last_episode, last_episode + 1000, 10)
                plt.plot(future_episodes, p(future_episodes), 'r:', alpha=0.5)
        
        # Add statistics
        eval_max = np.max(eval_rewards)
        eval_max_idx = np.argmax(eval_rewards)
        plt.plot(eval_episodes[eval_max_idx], eval_max, 'g*', markersize=10,
                 label=f'Best: {eval_max:.1f}')
        
        plt.axhline(y=np.mean(eval_rewards), color='k', linestyle='--',
                    label=f'Mean: {np.mean(eval_rewards):.1f}')
        
        # Show most recent evaluation with text
        if len(eval_rewards) > 0:
            plt.annotate(f'{eval_rewards[-1]:.1f}', 
                     (eval_episodes[-1], eval_rewards[-1]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    else:
        plt.text(0.5, 0.5, 'No Evaluation Data', ha='center', va='center', 
                 transform=plt.gca().transAxes)
    
    plt.xlabel('Episodes')
    plt.ylabel('Evaluation Reward')
    plt.title('Evaluation Performance')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    # 3. Plot value and policy losses
    plt.subplot(grid_size, 2, 3)
    if value_losses is not None and len(value_losses) > 0:
        updates = np.arange(1, len(value_losses) + 1)
        plt.plot(updates, value_losses, 'g-', alpha=0.7, label='Value Loss')
        
        # Add smoothed line for value loss
        window_size = min(len(value_losses) // 10, 20)
        if window_size > 1:
            smoothed_value_loss = pd.Series(value_losses).rolling(window=window_size).mean().values
            plt.plot(updates[window_size-1:], smoothed_value_loss[~np.isnan(smoothed_value_loss)], 
                     'g-', linewidth=2, label=f'Smoothed (w={window_size})')
    
    plt.xlabel('Updates')
    plt.ylabel('Value Loss')
    plt.title('Value Function Loss')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplot(grid_size, 2, 4)
    if policy_losses is not None and len(policy_losses) > 0:
        updates = np.arange(1, len(policy_losses) + 1)
        plt.plot(updates, policy_losses, 'b-', alpha=0.7, label='Policy Loss')
        
        # Add smoothed line for policy loss
        window_size = min(len(policy_losses) // 10, 20)
        if window_size > 1:
            smoothed_policy_loss = pd.Series(policy_losses).rolling(window=window_size).mean().values
            plt.plot(updates[window_size-1:], smoothed_policy_loss[~np.isnan(smoothed_policy_loss)], 
                     'b-', linewidth=2, label=f'Smoothed (w={window_size})')
    
    plt.xlabel('Updates')
    plt.ylabel('Policy Loss')
    plt.title('Policy Loss')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 4. Plot entropy and clip fraction/KL divergence
    plt.subplot(grid_size, 2, 5)
    if entropies is not None and len(entropies) > 0:
        updates = np.arange(1, len(entropies) + 1)
        plt.plot(updates, entropies, 'purple', alpha=0.7, label='Entropy')
        
        # Add smoothed line for entropy
        window_size = min(len(entropies) // 10, 20)
        if window_size > 1:
            smoothed_entropy = pd.Series(entropies).rolling(window=window_size).mean().values
            plt.plot(updates[window_size-1:], smoothed_entropy[~np.isnan(smoothed_entropy)], 
                     'purple', linewidth=2, label=f'Smoothed (w={window_size})')
    
    plt.xlabel('Updates')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplot(grid_size, 2, 6)
    if clip_fractions is not None and len(clip_fractions) > 0:
        updates = np.arange(1, len(clip_fractions) + 1)
        plt.plot(updates, clip_fractions, 'orange', alpha=0.7, label='Clip Fraction')
        
        # Add smoothed line for clip fraction
        window_size = min(len(clip_fractions) // 10, 20)
        if window_size > 1:
            smoothed_clip = pd.Series(clip_fractions).rolling(window=window_size).mean().values
            plt.plot(updates[window_size-1:], smoothed_clip[~np.isnan(smoothed_clip)], 
                    'orange', linewidth=2, label=f'Smoothed (w={window_size})')
        
        # Add warning thresholds
        plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Warning Threshold')
    
    plt.xlabel('Updates')
    plt.ylabel('Clip Fraction')
    plt.title('PPO Clip Fraction')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 5. Add KL divergence plot for PPO-KL
    if kl_divergences is not None and len(kl_divergences) > 0:
        plt.subplot(grid_size, 2, 7)
        updates = np.arange(1, len(kl_divergences) + 1)
        plt.plot(updates, kl_divergences, 'brown', alpha=0.7, label='KL Divergence')
        
        # Add smoothed line for KL
        window_size = min(len(kl_divergences) // 10, 20)
        if window_size > 1:
            smoothed_kl = pd.Series(kl_divergences).rolling(window=window_size).mean().values
            plt.plot(updates[window_size-1:], smoothed_kl[~np.isnan(smoothed_kl)], 
                     'brown', linewidth=2, label=f'Smoothed (w={window_size})')
        
        plt.xlabel('Updates')
        plt.ylabel('KL Divergence')
        plt.title('Policy KL Divergence')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add overall title
    plt.suptitle(f'{algorithm} Learning Curves', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    curves_dir = 'learning_curves'
    if not os.path.exists(curves_dir):
        os.makedirs(curves_dir)
        
    # Add timestamp to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = f"{curves_dir}/{filename}_{timestamp}.png"
    
    plt.savefig(full_path, dpi=120, bbox_inches='tight')
    print(f"Enhanced learning curve saved to {full_path}")
    
    plt.close()

def annealing_linear(start, end, pct):
    """
    Linear annealing from start to end as pct goes from 0.0 to 1.0
    """
    return start + pct * (end - start)

def test_env_agent_interaction(agent, vec_env, args):
    """Test the interaction between environment and agent"""
    print("\n=== Testing Environment-Agent Interaction ===")
    
    # Reset environment
    test_state, _ = vec_env.reset(seed=args.seed+999)
    print(f"Test state shape: {test_state.shape}")
    
    # Get action from agent
    test_action, test_log_prob, test_value = agent.select_actions_vec(test_state, evaluate=False)
    print(f"Test action shape: {test_action.shape}, type: {type(test_action)}")
    print(f"Test log prob shape: {test_log_prob.shape}")
    print(f"Test value shape: {test_value.shape}")
    
    # Try stepping the environment with the test action
    try:
        test_next_state, test_reward, test_term, test_trunc, test_info = vec_env.step(test_action)
        print("✓ Environment step successful!")
        print(f"Reward: {test_reward}")
        print(f"Next state shape: {test_next_state.shape}")
        return True
    except Exception as e:
        print(f"✗ ERROR stepping environment: {e}")
        return False
    finally:
        print("=== End of Interaction Test ===\n")

def collect_rollouts(agent, vec_env, rollouts, steps_per_update, num_envs, updates, verbose=False):
    """
    Collect rollout data from environments.
    
    Args:
        agent: PPO agent
        vec_env: Vectorized environment
        rollouts: RolloutStorage to store data
        steps_per_update: Steps to collect before updating
        num_envs: Number of parallel environments
        updates: Current update count
        verbose: Whether to print verbose output
        
    Returns:
        states: Final states after collection
        bool: Whether collection was successful
    """    
    # Reset rollout storage
    rollouts.clear()

    # Get initial states
    states, _ = vec_env.reset(seed=updates + 1000)
    
    # Track episode info
    episode_rewards = [0] * num_envs
    episode_steps = [0] * num_envs
    
    # Collect steps
    for step in range(steps_per_update):
        # Select actions
        actions, log_probs, values = agent.select_actions_vec(states, evaluate=False)
                
        # Ensure actions have the right shape for the environment
        if hasattr(vec_env, 'single_action_space') and hasattr(vec_env.single_action_space, 'shape'):
            if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                if isinstance(actions, np.ndarray) and actions.ndim == 2 and actions.shape[1] == 1:
                    actions_env = actions.flatten().astype(np.int32)
                else:
                    actions_env = actions.astype(np.int32)
            else:
                actions_env = actions
        else:
            actions_env = actions
            
        # Take steps in environments
        try:
            next_states, rewards, terminations, truncations, infos = vec_env.step(actions_env)
            
            # Combine terminations and truncations
            dones = np.logical_or(terminations, truncations)
            
            # Create mask (0 if done, 1 if not done)
            masks = 1.0 - dones.astype(np.float32)
            
            # Store transitions in rollout storage
            for i in range(num_envs):

                # Push to rollout
                rollouts.push(
                    states[i], 
                    actions[i], 
                    rewards[i], 
                    values[i], 
                    log_probs[i], 
                    masks[i]
                )
                
                # Update episode stats
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                
                # Handle episode termination
                if dones[i] and verbose:
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
            
            # Update states
            states = next_states
            
        except Exception as e:
            print(f"ERROR during environment step: {e}")
            return states, False
        
    # Verify rollout data
    if rollouts.position == 0:
        print("ERROR: No data collected in rollouts!")
        return states, False
    
    return states, True

def main():
    """Main training function for PPO with support for both PPOCLIP and PPOKL"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.algorithm}_{args.env_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # Auto-generate curve name if not provided
    if args.curve_name is None:
        args.curve_name = f"{args.algorithm}_{args.env_name}_learning_curve"
    
    # Configure device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        tqdm.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Set GPU memory limits if using CUDA
        if args.memory_efficient:
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    else:
        tqdm.write("Using CPU")
    
    # Create vectorized environments
    tqdm.write(f"Creating {args.num_envs} parallel environments for '{args.env_name}'")
    
    # Initialize render mode only if explicitly requested and only with a single environment
    render_mode = "human" if args.render and args.num_envs == 1 else None
    if args.render and args.num_envs > 1:
        tqdm.write("Warning: Rendering is only supported for single environment training. Disabling rendering.")
    
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
    action_space = vec_env.single_action_space
    
    # Handle different types of action spaces
    if isinstance(vec_env.single_action_space, gym.spaces.Box):
        action_dim = vec_env.single_action_space.shape[0]
        tqdm.write(f"Continuous action space detected with {action_dim} dimensions")
    elif isinstance(vec_env.single_action_space, gym.spaces.Discrete):
        action_dim = 1  # Discrete action is represented as a single integer
        tqdm.write(f"Discrete action space detected with {vec_env.single_action_space.n} possible actions")
    else:
        raise ValueError(f"Unsupported action space type: {type(vec_env.single_action_space)}")
        
    tqdm.write(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Initialize PPO agent based on algorithm choice
    if args.algorithm == "PPOCLIP":
        agent = PPOCLIP(state_dim, action_space, args)
        tqdm.write(f"Initialized PPOCLIP agent with clip parameter: {args.clip_param}")
    else:  # PPOKL
        agent = PPOKL(state_dim, action_space, args)
        tqdm.write(f"Initialized PPOKL agent with KL target: {args.kl_target}")
    
    # Set up TensorBoard writer with unique directory
    log_dir = f'runs/{args.experiment_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    writer = SummaryWriter(log_dir)
    tqdm.write(f"TensorBoard logs will be saved to: {log_dir}")
    tqdm.write(f"Run 'tensorboard --logdir={log_dir}' to view")
    
    # Create rollout storage based on number of steps per update
    steps_per_update = args.update_interval
    rollout_size = steps_per_update * args.num_envs  # Total transitions before update
    rollouts = RolloutStorage(rollout_size, state_dim, action_dim, args.seed)
    
    # Training loop parameters
    total_numsteps = 0
    updates = 0
    
    # To keep track of episodes in vectorized environment
    episode_rewards = [0] * args.num_envs
    episode_steps = [0] * args.num_envs
    episode_count = 0
    
    # For tracking average reward
    recent_rewards = []
    moving_avg_window = 10  # Window size for moving average
    avg_reward = 0.0
    
    # Lists to store metrics for learning curve
    all_episode_rewards = []
    all_episode_numbers = []
    all_eval_rewards = []
    all_eval_episode_numbers = []
    
    # Lists to store training metrics
    value_losses = []
    policy_losses = []
    entropies = []
    clip_fractions = []
    kl_divergences = []

    # Best evaluation reward for saving the best model
    best_eval = float('-inf')
    
    # Calculate total training duration
    total_updates = args.num_steps // (args.update_interval * args.num_envs)
    
    # Initialize progress bar for total training
    progress_bar = tqdm(total=args.num_steps, desc="Training Progress", unit="steps")
    
    # Reset environments to get initial states
    states, _ = vec_env.reset(seed=args.seed)
    
    # Print training configuration
    tqdm.write("\nTraining Configuration:")
    tqdm.write(f"  Algorithm: {args.algorithm}")
    tqdm.write(f"  Environment: {args.env_name} with {args.obstacle_shape} obstacles")
    tqdm.write(f"  Parallel Environments: {args.num_envs}")
    tqdm.write(f"  Steps per update: {args.update_interval}")
    tqdm.write(f"  Batch size: {args.batch_size}")
    tqdm.write(f"  PPO epochs: {args.ppo_epoch}")
    tqdm.write(f"  Learning rates - Policy: {args.policy_lr}, Value: {args.value_lr}")
    if args.algorithm == "PPOCLIP":
        tqdm.write(f"  Clip parameter: {args.clip_param}")
    else:
        tqdm.write(f"  KL target: {args.kl_target}")
    tqdm.write(f"  Normalize advantages: {args.normalize_advantages}")
    tqdm.write(f"  Learning rate annealing: {args.lr_annealing}")
    tqdm.write(f"  Total updates to perform: {total_updates}")
    tqdm.write("")
    
    try:
        while total_numsteps < args.num_steps:
            # Run garbage collection periodically
            if updates % args.gc_interval == 0 and updates > 0:
                collect_garbage()
            
            # Apply learning rate annealing if enabled
            if args.lr_annealing:
                # Linear annealing from 100% to 10% of original lr
                progress = min(1.0, total_numsteps / args.num_steps)
                
                # Calculate annealed learning rates
                policy_lr = args.policy_lr * (1.0 - 0.9 * progress)  # Linear from 100% to 10%
                value_lr = args.value_lr * (1.0 - 0.9 * progress)
                
                # Update learning rates
                for param_group in agent.policy_optimizer.param_groups:
                    param_group['lr'] = policy_lr
                for param_group in agent.value_optimizer.param_groups:
                    param_group['lr'] = value_lr
                
                # Log learning rate changes periodically
                if updates % 10 == 0:
                    writer.add_scalar('hyperparams/policy_lr', policy_lr, updates)
                    writer.add_scalar('hyperparams/value_lr', value_lr, updates)
            
            # Start a new rollout collection cycle
            rollouts.clear()
            
            # Collect rollout data
            for step in range(0, steps_per_update):
                # Select actions
                actions, log_probs, values = agent.select_actions_vec(states, evaluate=False)
                                
                # Ensure actions have the right shape for the environment
                if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                    if isinstance(actions, np.ndarray) and actions.ndim == 2 and actions.shape[1] == 1:
                        actions_env = actions.flatten().astype(np.int32)
                    else:
                        actions_env = actions.astype(np.int32)
                else:
                    actions_env = actions
                
                try:
                    # Take steps in environments
                    next_states, rewards, terminations, truncations, infos = vec_env.step(actions_env)
                    
                    # Combine terminations and truncations
                    dones = np.logical_or(terminations, truncations)
                    
                    # Create mask (0 if done, 1 if not done)
                    masks = 1.0 - dones.astype(np.float32)
                    
                    # Store transitions in rollout storage
                    for i in range(args.num_envs):
                        
                        # Push to rollout storage
                        rollouts.push(
                            states[i], 
                            actions[i], 
                            rewards[i], 
                            values[i], 
                            log_probs[i], 
                            masks[i]
                        )
                        
                        # Update episode statistics
                        episode_rewards[i] += rewards[i]
                        episode_steps[i] += 1
                        
                        # Handle episode termination
                        if dones[i]:
                            # Log episode stats
                            episode_count += 1
                            writer.add_scalar('train/episode_reward', episode_rewards[i], episode_count)
                            writer.add_scalar('train/episode_length', episode_steps[i], episode_count)
                            writer.add_scalar('train/reward_per_step', episode_rewards[i] / episode_steps[i], episode_count)
                            
                            # Store training metrics for learning curve
                            all_episode_rewards.append(episode_rewards[i])
                            all_episode_numbers.append(episode_count)
                            
                            # Update recent rewards list and calculate moving average
                            recent_rewards.append(episode_rewards[i])
                            if len(recent_rewards) > moving_avg_window:
                                recent_rewards.pop(0)  # Remove oldest reward
                            avg_reward = sum(recent_rewards) / len(recent_rewards)
                            writer.add_scalar('train/avg_reward', avg_reward, episode_count)
                            
                            # Print progress if verbose
                            if args.verbose:
                                tqdm.write(f"Episode {episode_count}: Reward={episode_rewards[i]:.2f}, "
                                        f"Steps={episode_steps[i]}, "
                                        f"Reward/Step={episode_rewards[i]/episode_steps[i]:.4f}")
                            
                            # Reset episode stats
                            episode_rewards[i] = 0
                            episode_steps[i] = 0
                    
                    # Update states
                    states = next_states
                    
                    # Update progress bar after each step
                    total_numsteps += args.num_envs
                    progress_bar.update(args.num_envs)
                    
                    # Update progress bar display details (less frequently to reduce overhead)
                    if step % 5 == 0:
                        progress_bar.set_postfix({
                            'episodes': episode_count,
                            'updates': updates,
                            'avg_reward': f"{avg_reward:.2f}"
                        })
                    
                except Exception as e:
                    tqdm.write(f"ERROR during environment step: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                        
            # Check if we have enough data
            if rollouts.position == 0:
                tqdm.write("ERROR: No data collected in rollouts! Skipping update.")
                updates += 1
                continue
                        
            # Get final value estimates for computing advantages
            with torch.no_grad():
                final_values = agent.get_value(states)
            
            # Compute returns and advantages
            rollouts.compute_returns(final_values, args.gamma, args.tau)
            
            # Verify rollout data before updating
            rollout_data = rollouts.get_data()
            if len(rollout_data.get('states', [])) == 0:
                tqdm.write("Warning: Empty rollout data after computing returns. Skipping update.")
                updates += 1
                continue
                        
            # Perform PPO update based on algorithm choice
            if args.algorithm == "PPOCLIP":
                value_loss, policy_loss, entropy_loss, clip_fraction = agent.update_parameters(rollouts)
                
                # Store metrics for learning curve
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
                entropies.append(entropy_loss)
                clip_fractions.append(clip_fraction)
                
                # Log detailed metrics to tensorboard
                writer.add_scalar('train/value_loss', value_loss, updates)
                writer.add_scalar('train/policy_loss', policy_loss, updates)
                writer.add_scalar('train/entropy', entropy_loss, updates)
                writer.add_scalar('train/clip_fraction', clip_fraction, updates)
                
                # Log histogram of advantages
                if updates % 10 == 0:
                    try:
                        advantages = rollout_data['advantages']
                        if len(advantages) > 0:
                            writer.add_histogram('train/advantages', advantages, updates)
                    except Exception as e:
                        tqdm.write(f"Error logging histogram: {e}")
                
                # Print update stats if verbose
                if args.verbose or updates % args.log_interval == 0:
                    tqdm.write(f"Update {updates}/{total_updates}: "
                             f"Policy Loss={policy_loss:.5f}, Value Loss={value_loss:.5f}, "
                             f"Entropy={entropy_loss:.5f}, Clip Fraction={clip_fraction:.3f}")
            else:  # PPOKL
                value_loss, policy_loss, entropy_loss, kl_divergence, kl_beta = agent.update_parameters(rollouts)
                
                # Store metrics for learning curve
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
                entropies.append(entropy_loss)
                kl_divergences.append(kl_divergence)
                
                # Log detailed metrics to tensorboard
                writer.add_scalar('train/value_loss', value_loss, updates)
                writer.add_scalar('train/policy_loss', policy_loss, updates)
                writer.add_scalar('train/entropy', entropy_loss, updates)
                writer.add_scalar('train/kl_divergence', kl_divergence, updates)
                writer.add_scalar('train/kl_beta', kl_beta, updates)
                
                # Print update stats if verbose
                if args.verbose or updates % args.log_interval == 0:
                    tqdm.write(f"Update {updates}/{total_updates}: "
                             f"Policy Loss={policy_loss:.5f}, Value Loss={value_loss:.5f}, "
                             f"Entropy={entropy_loss:.5f}, KL={kl_divergence:.5f}, KL Coef={kl_beta:.3f}")
            
            # Update progress bar display after update (not incrementing steps again)
            progress_bar.set_postfix({
                'episodes': episode_count,
                'updates': updates,
                'avg_reward': f"{avg_reward:.2f}"
            })
            
            # Increment update counter
            updates += 1

            # Run evaluation periodically
            if updates % args.eval_interval == 0:
                # Set evaluation message
                progress_bar.set_description("Evaluating...")
                
                # Run evaluation
                eval_mean_reward, eval_std_reward = evaluate_policy_vec(
                    agent, eval_env, max_eval_episodes=args.eval_episodes)
                
                if eval_mean_reward > best_eval:
                    best_eval = eval_mean_reward
                    # Save best model checkpoint
                    agent.save_checkpoint(args.env_name, suffix=f"{args.experiment_name}_best")
                    
                    tqdm.write(f"New best evaluation reward: {best_eval:.2f} saved as {args.experiment_name}_best.pt")
                
                # Log evaluation metrics
                writer.add_scalar('eval/mean_reward', eval_mean_reward, updates)
                writer.add_scalar('eval/std_reward', eval_std_reward, updates)
                all_eval_rewards.append(eval_mean_reward)
                all_eval_episode_numbers.append(episode_count)
                
                # Print evaluation results
                tqdm.write(f"Evaluation after update {updates}/{total_updates}: " 
                         f"Mean Reward: {eval_mean_reward:.2f}, "
                         f"Std: {eval_std_reward:.2f}")
                
                # Reset progress bar description
                progress_bar.set_description("Training Progress")
                
                # Run garbage collection after evaluation
                collect_garbage()
            
            # Save model checkpoint periodically
            if updates % args.checkpoint_interval == 0:
                checkpoint_path = f"checkpoints/{args.algorithm}_{args.env_name}_update_{updates}.pt"
                agent.save_checkpoint(args.env_name, suffix=f"{args.experiment_name}_update_{updates}")
                
                # Also save rollouts occasionally for potential resume
                if updates % (args.checkpoint_interval * 5) == 0:
                    rollouts.save_rollouts(args.env_name, suffix=f"{args.experiment_name}_update_{updates}")
                
                # Log checkpoint info
                tqdm.write(f"Checkpoint saved at update {updates} to {checkpoint_path}")
    
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user")
    except Exception as e:
        tqdm.write(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close progress bar
        progress_bar.close()
        
        # Final save
        if 'updates' in locals() and updates > 0:
            tqdm.write(f"Saving final checkpoint at update {updates}")
            agent.save_checkpoint(args.env_name, suffix=f"{args.experiment_name}_final_{updates}")
        
        # Clean up
        writer.close()
        vec_env.close()
        eval_env.close()
        
        # Save enhanced learning curve if requested
        if args.save_curve and all_episode_numbers:
            try:
                # Set algorithm name based on variant
                if args.algorithm == "PPOCLIP":
                    alg_name = f"PPO-CLIP (clip={args.clip_param})"
                    save_enhanced_learning_curve(
                        all_episode_numbers, all_episode_rewards,
                        all_eval_episode_numbers, all_eval_rewards,
                        value_losses, policy_losses, entropies, clip_fractions, None,
                        f"{args.curve_name}_{args.experiment_name}", alg_name)
                else:  # PPOKL
                    alg_name = f"PPO-KL (target={args.kl_target})"
                    save_enhanced_learning_curve(
                        all_episode_numbers, all_episode_rewards,
                        all_eval_episode_numbers, all_eval_rewards,
                        value_losses, policy_losses, entropies, None, kl_divergences,
                        f"{args.curve_name}_{args.experiment_name}", alg_name)
            except Exception as e:
                tqdm.write(f"Error saving learning curve: {e}")
        
        # Final cleanup
        collect_garbage()
        tqdm.write("Training finished")
        
if __name__ == "__main__":
    # Set lower memory usage for numpy operations
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    
    # Set CUDA options to optimize memory usage
    if torch.cuda.is_available():
        # Try to ensure PyTorch releases memory after operations
        torch.cuda.empty_cache()
        
    # Run main function
    main()