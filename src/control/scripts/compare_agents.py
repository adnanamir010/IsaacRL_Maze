#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import datetime
import os
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import json

# Import from your existing modules
from agents import SAC, PPOCLIP
from memory import ReplayMemory, RolloutStorage
from rl_utils import evaluate_policy, evaluate_policy_vec, save_learning_curve, collect_garbage
from environment import VectorizedDDEnv, make_vectorized_env

def parse_arguments():
    """Parse command line arguments for the algorithm comparison script"""
    parser = argparse.ArgumentParser(description='Compare PPO-CLIP and SAC Algorithms')
    
    # Environment settings
    parser.add_argument('--env-name', default="VectorizedDD",
                    help='Environment name (default: VectorizedDD)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--num-steps', type=int, default=500_000, metavar='N',
                    help='Maximum number of steps (default: 500_000)')
    parser.add_argument('--num-envs', type=int, default=4,
                    help='Number of parallel environments (default: 4)')
    parser.add_argument('--hidden-size', type=int, default=128, metavar='N',
                    help='Hidden size for both algorithms (default: 128)')
    
    # Shared algorithm parameters
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for both algorithms (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='Target smoothing coefficient for both algorithms (default: 0.005)')
    parser.add_argument('--cuda', action="store_true", default=True,
                    help='Run on CUDA if available (default: True)')
    
    # PPO-specific parameters
    parser.add_argument('--ppo-clip-param', type=float, default=0.2, metavar='G',
                    help='PPO clip parameter (default: 0.2)')
    parser.add_argument('--ppo-epoch', type=int, default=10, metavar='G',
                    help='Number of PPO epochs (default: 10)')
    parser.add_argument('--num-mini-batch', type=int, default=32, metavar='G',
                    help='Number of PPO mini-batches (default: 32)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, metavar='G',
                    help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, metavar='G',
                    help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, metavar='G',
                    help='Max gradient norm (default: 0.5)')
    parser.add_argument('--use-clipped-value-loss', action='store_true', default=True,
                    help='Use clipped value loss (default: True)')
    parser.add_argument('--policy-lr', type=float, default=3e-4, metavar='G',
                    help='Policy learning rate for PPO (default: 3e-4)')
    parser.add_argument('--value-lr', type=float, default=1e-3, metavar='G',
                    help='Value function learning rate for PPO (default: 1e-3)')
    parser.add_argument('--ppo-batch-size', type=int, default=2048, metavar='N',
                    help='Batch size for PPO updates (default: 2048)')
    parser.add_argument('--update-interval', type=int, default=2048, metavar='N',
                    help='Steps between PPO updates (default: 2048)')
    
    # SAC-specific parameters
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type for SAC: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α for entropy in SAC (default: 0.2)')
    parser.add_argument('--automatic-entropy-tuning', type=bool, default=True, metavar='G',
                    help='Automatically adjust α in SAC (default: True)')
    parser.add_argument('--sac-lr', type=float, default=0.0003, metavar='G',
                    help='Learning rate for SAC (default: 0.0003)')
    parser.add_argument('--sac-batch-size', type=int, default=256, metavar='N',
                    help='Batch size for SAC (default: 256)')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                    help='Model updates per simulator step for SAC (default: 1)')
    parser.add_argument('--start-steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions in SAC (default: 10000)')
    parser.add_argument('--target-update-interval', type=int, default=10, metavar='N',
                    help='Value target update interval for SAC (default: 10)')
    parser.add_argument('--replay-size', type=int, default=500_000, metavar='N',
                    help='Size of replay buffer for SAC (default: 500_000)')
    
    # Evaluation settings
    parser.add_argument('--eval-episodes', type=int, default=10,
                    help='Number of episodes for evaluation (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=5000,
                    help='Step interval between evaluations (default: 5000)')
    parser.add_argument('--render', action="store_true",
                    help='Render visualization (default: False)')
    
    # Output settings
    parser.add_argument('--log-interval', type=int, default=1000,
                    help='Logging interval in steps (default: 1000)')
    parser.add_argument('--checkpoint-interval', type=int, default=20000,
                    help='Checkpoint save interval in steps (default: 20000)')
    parser.add_argument('--output-dir', type=str, default='results',
                    help='Directory to save results (default: results)')
    parser.add_argument('--run-name', type=str, default=None,
                    help='Custom name for this run (default: timestamp)')
    
    return parser.parse_args()

def train_ppo(args, vec_env, eval_env, writer, device, common_metrics, log_dir):
    """
    Train a PPO-CLIP agent and collect metrics.
    
    Args:
        args: Command line arguments
        vec_env: Vectorized environment for training
        eval_env: Environment for evaluation
        writer: TensorBoard writer
        device: Device to run on (CPU/GPU)
        common_metrics: Dict to store metrics for comparison
        log_dir: Directory to save logs
        
    Returns:
        PPOCLIP: Trained agent
    """
    print("\n=== Starting PPO-CLIP Training ===")
    
    # Get environment dimensions
    state_dim = vec_env.single_observation_space.shape[0]
    action_space = vec_env.single_action_space
    
    # Initialize PPO-CLIP agent
    agent = PPOCLIP(state_dim, action_space, args)
    
    # Create rollout storage
    steps_per_update = args.update_interval
    rollout_size = steps_per_update * args.num_envs
    rollouts = RolloutStorage(rollout_size, state_dim, 
                             action_dim=1 if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0], 
                             seed=args.seed)
    
    # Training loop parameters
    total_numsteps = 0
    updates = 0
    start_time = time.time()
    
    # Track metrics
    ppo_metrics = {
        'steps': [],
        'train_rewards': [],
        'eval_mean_rewards': [],
        'eval_std_rewards': [],
        'value_losses': [],
        'policy_losses': [],
        'entropy_losses': [],
        'clip_fractions': [],
        'times': []
    }
    
    # To keep track of episodes
    episode_rewards = [0] * args.num_envs
    episode_steps = [0] * args.num_envs
    episode_count = 0
    
    # Initialize progress bar
    progress_bar = tqdm(total=args.num_steps, desc="PPO-CLIP Training", unit="steps")
    
    # Reset environments
    states, _ = vec_env.reset(seed=args.seed)
    
    while total_numsteps < args.num_steps:
        # Clear rollout storage for new collection
        rollouts.clear()
        
        # Collect rollout data
        for step in range(0, steps_per_update):
            # Select actions
            actions, log_probs, values = agent.select_actions_vec(states, evaluate=False)
            
            # Ensure actions have the right shape
            if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                if isinstance(actions, np.ndarray) and actions.ndim == 2 and actions.shape[1] == 1:
                    actions_env = actions.flatten().astype(np.int32)
                else:
                    actions_env = actions.astype(np.int32)
            else:
                actions_env = actions
            
            # Take steps in environments
            next_states, rewards, terminations, truncations, infos = vec_env.step(actions_env)
            
            # Combine terminations and truncations
            dones = np.logical_or(terminations, truncations)
            
            # Create mask (0 if done, 1 if not done)
            masks = 1.0 - dones.astype(np.float32)
            
            # Store transitions in rollout storage
            for i in range(args.num_envs):
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
                    writer.add_scalar('PPO/reward/train', episode_rewards[i], total_numsteps)
                    ppo_metrics['train_rewards'].append((total_numsteps, episode_rewards[i]))
                    
                    # Reset episode stats
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
            
            # Update states
            states = next_states
            total_numsteps += args.num_envs
            
            # Update progress bar
            progress_bar.update(args.num_envs)
            progress_bar.set_postfix({
                'episodes': episode_count,
                'steps': total_numsteps
            })
            
            # Evaluate periodically
            if total_numsteps % args.eval_interval <= args.num_envs:
                eval_mean_reward, eval_std_reward = evaluate_policy_vec(
                    agent, eval_env, max_eval_episodes=args.eval_episodes)
                
                writer.add_scalar('PPO/reward/eval', eval_mean_reward, total_numsteps)
                
                # Save metrics
                ppo_metrics['eval_mean_rewards'].append((total_numsteps, eval_mean_reward))
                ppo_metrics['eval_std_rewards'].append((total_numsteps, eval_std_reward))
                ppo_metrics['steps'].append(total_numsteps)
                ppo_metrics['times'].append(time.time() - start_time)
                
                # Add to common metrics for comparison
                if total_numsteps not in common_metrics['steps']:
                    common_metrics['steps'].append(total_numsteps)
                
                common_metrics['ppo_mean_rewards'][total_numsteps] = eval_mean_reward
                common_metrics['ppo_std_rewards'][total_numsteps] = eval_std_reward
                
                # Log metrics
                tqdm.write(f"PPO-CLIP steps: {total_numsteps}, mean reward: {eval_mean_reward:.2f}, std: {eval_std_reward:.2f}")
        
        # Get final value estimates for computing advantages
        with torch.no_grad():
            final_values = agent.get_value(states)
            
        # Compute returns and advantages using GAE
        rollouts.compute_returns(final_values, args.gamma, args.tau)
        
        # Perform PPO update
        value_loss, policy_loss, entropy_loss, clip_fraction = agent.update_parameters(rollouts)
        
        # Log update statistics
        updates += 1
        if total_numsteps % args.log_interval <= args.num_envs:
            writer.add_scalar('PPO/loss/value', value_loss, total_numsteps)
            writer.add_scalar('PPO/loss/policy', policy_loss, total_numsteps)
            writer.add_scalar('PPO/loss/entropy', entropy_loss, total_numsteps)
            writer.add_scalar('PPO/stats/clip_fraction', clip_fraction, total_numsteps)
            
            # Save metrics
            ppo_metrics['value_losses'].append((total_numsteps, value_loss))
            ppo_metrics['policy_losses'].append((total_numsteps, policy_loss))
            ppo_metrics['entropy_losses'].append((total_numsteps, entropy_loss))
            ppo_metrics['clip_fractions'].append((total_numsteps, clip_fraction))
        
        # Save checkpoint periodically
        if total_numsteps % args.checkpoint_interval <= args.num_envs:
            checkpoint_path = os.path.join(log_dir, f"ppo_checkpoint_{total_numsteps}.pt")
            agent.save_checkpoint(args.env_name, suffix=f"step_{total_numsteps}", ckpt_path=checkpoint_path)
    
    # Close progress bar
    progress_bar.close()
    
    # Save final metrics
    metrics_path = os.path.join(log_dir, "ppo_metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy values to native Python types
        serializable_metrics = {
            'steps': [int(step) for step in ppo_metrics['steps']],
            'train_rewards': [(int(step), float(reward)) for step, reward in ppo_metrics['train_rewards']],
            'eval_mean_rewards': [(int(step), float(reward)) for step, reward in ppo_metrics['eval_mean_rewards']],
            'eval_std_rewards': [(int(step), float(std)) for step, std in ppo_metrics['eval_std_rewards']],
            'value_losses': [(int(step), float(loss)) for step, loss in ppo_metrics['value_losses']],
            'policy_losses': [(int(step), float(loss)) for step, loss in ppo_metrics['policy_losses']],
            'entropy_losses': [(int(step), float(loss)) for step, loss in ppo_metrics['entropy_losses']],
            'clip_fractions': [(int(step), float(frac)) for step, frac in ppo_metrics['clip_fractions']],
            'times': [float(t) for t in ppo_metrics['times']]
        }
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"PPO-CLIP training completed. Final evaluation reward: {ppo_metrics['eval_mean_rewards'][-1][1]:.2f}")
    
    return agent, ppo_metrics

def train_sac(args, vec_env, eval_env, writer, device, common_metrics, log_dir):
    """
    Train a SAC agent and collect metrics.
    
    Args:
        args: Command line arguments
        vec_env: Vectorized environment for training
        eval_env: Environment for evaluation
        writer: TensorBoard writer
        device: Device to run on (CPU/GPU)
        common_metrics: Dict to store metrics for comparison
        log_dir: Directory to save logs
        
    Returns:
        SAC: Trained agent
    """
    print("\n=== Starting SAC Training ===")
    
    # Get environment dimensions
    state_dim = vec_env.single_observation_space.shape[0]
    action_space = vec_env.single_action_space
    
    # Initialize SAC agent with appropriate parameters
    class SACArgs:
        def __init__(self, args):
            self.gamma = args.gamma
            self.tau = args.tau
            self.lr = args.sac_lr
            self.alpha = args.alpha
            self.policy = args.policy
            self.target_update_interval = args.target_update_interval
            self.hidden_size = args.hidden_size
            self.automatic_entropy_tuning = args.automatic_entropy_tuning
            self.cuda = args.cuda
    
    sac_args = SACArgs(args)
    agent = SAC(state_dim, action_space, sac_args)
    
    # Create replay memory
    memory = ReplayMemory(args.replay_size, state_dim, 
                         action_dim=1 if isinstance(action_space, gym.spaces.Discrete) else action_space.shape[0], 
                         seed=args.seed)
    
    # Training loop parameters
    total_numsteps = 0
    updates = 0
    start_time = time.time()
    
    # Track metrics
    sac_metrics = {
        'steps': [],
        'train_rewards': [],
        'eval_mean_rewards': [],
        'eval_std_rewards': [],
        'critic_1_losses': [],
        'critic_2_losses': [],
        'policy_losses': [],
        'alpha_losses': [],
        'alphas': [],
        'times': []
    }
    
    # To keep track of episodes
    episode_rewards = [0] * args.num_envs
    episode_steps = [0] * args.num_envs
    episode_count = 0
    
    # Initialize progress bar
    progress_bar = tqdm(total=args.num_steps, desc="SAC Training", unit="steps")
    
    # Reset environments
    states, _ = vec_env.reset(seed=args.seed)
    
    while total_numsteps < args.num_steps:
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
                    actions_env = actions.flatten().astype(np.int32)
                elif actions.ndim == 1:
                    # If shape is (n_envs,)
                    actions_env = actions.astype(np.int32)
            else:
                actions_env = actions
        else:
            actions_env = actions
        
        # Take steps in environments
        next_states, rewards, terminations, truncations, infos = vec_env.step(actions_env)
        
        # Combine terminations and truncations
        dones = np.logical_or(terminations, truncations)
        
        # Collect transitions in replay buffer
        for i in range(args.num_envs):
            # Handle the "done" signal properly
            terminal_done = dones[i]
            mask = 0.0 if terminal_done else 1.0  # Terminal state = 0, non-terminal = 1
            
            # Store transition in replay memory
            if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                # Convert discrete action to array form
                discrete_action = actions[i] if isinstance(actions, np.ndarray) else actions
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
                writer.add_scalar('SAC/reward/train', episode_rewards[i], total_numsteps)
                sac_metrics['train_rewards'].append((total_numsteps, episode_rewards[i]))
                
                # Reset episode stats
                episode_rewards[i] = 0
                episode_steps[i] = 0
        
        # Perform updates if enough samples in memory
        if len(memory) > args.sac_batch_size:
            # Number of updates per step
            for _ in range(args.updates_per_step * args.num_envs):
                # Update parameters
                critic_1_loss, critic_2_loss, policy_loss, alpha_loss, alpha = agent.update_parameters(
                    memory, args.sac_batch_size, updates)
                
                # Log metrics periodically to reduce overhead
                if total_numsteps % args.log_interval <= args.num_envs:
                    writer.add_scalar('SAC/loss/critic_1', critic_1_loss, total_numsteps)
                    writer.add_scalar('SAC/loss/critic_2', critic_2_loss, total_numsteps)
                    writer.add_scalar('SAC/loss/policy', policy_loss, total_numsteps)
                    writer.add_scalar('SAC/loss/alpha', alpha_loss, total_numsteps)
                    writer.add_scalar('SAC/parameters/alpha', alpha, total_numsteps)
                    
                    # Save metrics
                    sac_metrics['critic_1_losses'].append((total_numsteps, critic_1_loss))
                    sac_metrics['critic_2_losses'].append((total_numsteps, critic_2_loss))
                    sac_metrics['policy_losses'].append((total_numsteps, policy_loss))
                    sac_metrics['alpha_losses'].append((total_numsteps, alpha_loss))
                    sac_metrics['alphas'].append((total_numsteps, alpha))
                
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
        
        # Evaluate periodically
        if total_numsteps % args.eval_interval <= args.num_envs:
            eval_mean_reward, eval_std_reward = evaluate_policy_vec(
                agent, eval_env, max_eval_episodes=args.eval_episodes)
            
            writer.add_scalar('SAC/reward/eval', eval_mean_reward, total_numsteps)
            
            # Save metrics
            sac_metrics['eval_mean_rewards'].append((total_numsteps, eval_mean_reward))
            sac_metrics['eval_std_rewards'].append((total_numsteps, eval_std_reward))
            sac_metrics['steps'].append(total_numsteps)
            sac_metrics['times'].append(time.time() - start_time)
            
            # Add to common metrics for comparison
            if total_numsteps not in common_metrics['steps']:
                common_metrics['steps'].append(total_numsteps)
            
            common_metrics['sac_mean_rewards'][total_numsteps] = eval_mean_reward
            common_metrics['sac_std_rewards'][total_numsteps] = eval_std_reward
            
            # Log metrics
            tqdm.write(f"SAC steps: {total_numsteps}, mean reward: {eval_mean_reward:.2f}, std: {eval_std_reward:.2f}")
        
        # Save checkpoint periodically
        if total_numsteps % args.checkpoint_interval <= args.num_envs:
            checkpoint_path = os.path.join(log_dir, f"sac_checkpoint_{total_numsteps}.pt")
            agent.save_checkpoint(args.env_name, suffix=f"step_{total_numsteps}", ckpt_path=checkpoint_path)
    
    # Close progress bar
    progress_bar.close()
    
    # Save final metrics
    metrics_path = os.path.join(log_dir, "sac_metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy values to native Python types
        serializable_metrics = {
            'steps': [int(step) for step in sac_metrics['steps']],
            'train_rewards': [(int(step), float(reward)) for step, reward in sac_metrics['train_rewards']],
            'eval_mean_rewards': [(int(step), float(reward)) for step, reward in sac_metrics['eval_mean_rewards']],
            'eval_std_rewards': [(int(step), float(std)) for step, std in sac_metrics['eval_std_rewards']],
            'critic_1_losses': [(int(step), float(loss)) for step, loss in sac_metrics['critic_1_losses']],
            'critic_2_losses': [(int(step), float(loss)) for step, loss in sac_metrics['critic_2_losses']],
            'policy_losses': [(int(step), float(loss)) for step, loss in sac_metrics['policy_losses']],
            'alpha_losses': [(int(step), float(loss)) for step, loss in sac_metrics['alpha_losses']],
            'alphas': [(int(step), float(alpha)) for step, alpha in sac_metrics['alphas']],
            'times': [float(t) for t in sac_metrics['times']]
        }
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"SAC training completed. Final evaluation reward: {sac_metrics['eval_mean_rewards'][-1][1]:.2f}")
    
    return agent, sac_metrics

def plot_comparison(common_metrics, log_dir):
    """
    Create comparison plots for both algorithms.
    
    Args:
        common_metrics: Dict containing metrics from both algorithms
        log_dir: Directory to save plots
    """
    print("\n=== Creating Comparison Plots ===")
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Sort steps for consistent plotting
    steps = sorted(common_metrics['steps'])
    
    # Extract metrics for each algorithm
    ppo_rewards = [common_metrics['ppo_mean_rewards'][step] if step in common_metrics['ppo_mean_rewards'] else None for step in steps]
    ppo_stds = [common_metrics['ppo_std_rewards'][step] if step in common_metrics['ppo_std_rewards'] else None for step in steps]
    
    sac_rewards = [common_metrics['sac_mean_rewards'][step] if step in common_metrics['sac_mean_rewards'] else None for step in steps]
    sac_stds = [common_metrics['sac_std_rewards'][step] if step in common_metrics['sac_std_rewards'] else None for step in steps]
    
    # Filter out None values
    ppo_data = [(s, r, std) for s, r, std in zip(steps, ppo_rewards, ppo_stds) if r is not None]
    sac_data = [(s, r, std) for s, r, std in zip(steps, sac_rewards, sac_stds) if r is not None]
    
    if not ppo_data or not sac_data:
        print("Warning: Not enough data to create comparison plots")
        return
    
    # Extract data for plotting
    ppo_steps, ppo_rewards, ppo_stds = zip(*ppo_data)
    sac_steps, sac_rewards, sac_stds = zip(*sac_data)
    
    # Convert to numpy arrays
    ppo_steps = np.array(ppo_steps)
    ppo_rewards = np.array(ppo_rewards)
    ppo_stds = np.array(ppo_stds)
    
    sac_steps = np.array(sac_steps)
    sac_rewards = np.array(sac_rewards)
    sac_stds = np.array(sac_stds)
    
    # 1. Learning Curves Comparison
    plt.figure(figsize=(10, 6))
    
    # Plot PPO-CLIP
    plt.plot(ppo_steps, ppo_rewards, 'b-', label='PPO-CLIP', linewidth=2)
    plt.fill_between(ppo_steps, 
                    ppo_rewards - ppo_stds, 
                    ppo_rewards + ppo_stds, 
                    color='b', alpha=0.2)
    
    # Plot SAC
    plt.plot(sac_steps, sac_rewards, 'r-', label='SAC', linewidth=2)
    plt.fill_between(sac_steps, 
                    sac_rewards - sac_stds, 
                    sac_rewards + sac_stds, 
                    color='r', alpha=0.2)
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Mean Evaluation Reward')
    plt.title('PPO-CLIP vs SAC Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'learning_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sample Efficiency Comparison
    # Find the median of final rewards for both algorithms
    final_steps = min(ppo_steps[-1], sac_steps[-1])
    
    # Find rewards at comparable steps
    ppo_final_idx = np.argmin(np.abs(np.array(ppo_steps) - final_steps))
    sac_final_idx = np.argmin(np.abs(np.array(sac_steps) - final_steps))
    
    plt.figure(figsize=(10, 6))
    
    # Plot normalized learning progress
    ppo_normalized = ppo_rewards / ppo_rewards[ppo_final_idx]
    sac_normalized = sac_rewards / sac_rewards[sac_final_idx]
    
    plt.plot(ppo_steps, ppo_normalized, 'b-', label='PPO-CLIP', linewidth=2)
    plt.plot(sac_steps, sac_normalized, 'r-', label='SAC', linewidth=2)
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Normalized Performance')
    plt.title('Sample Efficiency Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'sample_efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart of final performance
    plt.figure(figsize=(8, 6))
    
    algorithms = ['PPO-CLIP', 'SAC']
    final_rewards = [ppo_rewards[-1], sac_rewards[-1]]
    final_stds = [ppo_stds[-1], sac_stds[-1]]
    
    # Create bars
    bars = plt.bar(algorithms, final_rewards, yerr=final_stds, capsize=10, 
                  color=['blue', 'red'], alpha=0.7)
    
    # Add value labels on top of bars
    for bar, reward in zip(bars, final_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{reward:.2f}", ha='center', va='bottom')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Final Mean Reward')
    plt.title('Final Performance Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(plots_dir, 'final_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save the metrics as CSV for further analysis
    import pandas as pd
    
    # Create DataFrame for PPO data
    ppo_df = pd.DataFrame({
        'steps': ppo_steps,
        'algorithm': 'PPO-CLIP',
        'mean_reward': ppo_rewards,
        'std_reward': ppo_stds
    })
    
    # Create DataFrame for SAC data
    sac_df = pd.DataFrame({
        'steps': sac_steps,
        'algorithm': 'SAC',
        'mean_reward': sac_rewards,
        'std_reward': sac_stds
    })
    
    # Combine DataFrames
    combined_df = pd.concat([ppo_df, sac_df], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(os.path.join(log_dir, 'comparison_metrics.csv'), index=False)
    
    print(f"Comparison plots saved to {plots_dir}")

def main():
    """Main function to run the algorithm comparison"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"{args.env_name}_{timestamp}"
    log_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)
    
    # Create environment factory for fair comparison
    if args.env_name == "VectorizedDD":
        env_factory = lambda: make_vectorized_env(num_envs=args.num_envs, seed=args.seed)
        eval_env_factory = lambda: make_vectorized_env(num_envs=1, seed=args.seed + 100)
    else:
        env_factory = lambda: gym.vector.make(args.env_name, num_envs=args.num_envs, asynchronous=False)
        eval_env_factory = lambda: gym.vector.make(args.env_name, num_envs=1, asynchronous=False)
    
    # Common metrics dictionary for comparison
    common_metrics = {
        'steps': [],
        'ppo_mean_rewards': {},
        'ppo_std_rewards': {},
        'sac_mean_rewards': {},
        'sac_std_rewards': {}
    }
    
    # Train PPO-CLIP
    vec_env = env_factory()
    eval_env = eval_env_factory()
    ppo_agent, ppo_metrics = train_ppo(args, vec_env, eval_env, writer, device, common_metrics, log_dir)
    vec_env.close()
    eval_env.close()
    collect_garbage()  # Clean up memory
    
    # Train SAC
    vec_env = env_factory()
    eval_env = eval_env_factory()
    sac_agent, sac_metrics = train_sac(args, vec_env, eval_env, writer, device, common_metrics, log_dir)
    vec_env.close()
    eval_env.close()
    
    # Create comparison plots
    plot_comparison(common_metrics, log_dir)
    
    # Clean up
    writer.close()
    collect_garbage()
    
    print(f"\n=== Comparison Complete ===")
    print(f"All results saved to {log_dir}")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={tensorboard_dir}")

if __name__ == "__main__":
    main()