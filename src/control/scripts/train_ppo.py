#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import datetime
import os
import gc
import gymnasium as gym
from tqdm import tqdm

from agents import PPOCLIP
from memory import RolloutStorage
from rl_utils import evaluate_policy, evaluate_policy_vec, save_learning_curve, collect_garbage
from environment import VectorizedDDEnv, make_vectorized_env
from torch.utils.tensorboard import SummaryWriter

def parse_arguments():
    """Parse command line arguments for the PPO-CLIP algorithm"""
    parser = argparse.ArgumentParser(description='PyTorch PPO-CLIP Args')
    parser.add_argument('--env-name', default="VectorizedDD",
                    help='Environment name (default: VectorizedDD)')
    parser.add_argument('--obstacle-shape', default="square",
                help='Obstacle shape: circular | square (default: square)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='GAE parameter (default: 0.95)')
    parser.add_argument('--policy-lr', type=float, default=3e-4, metavar='G',
                    help='Policy learning rate (default: 3e-4)')
    parser.add_argument('--value-lr', type=float, default=1e-3, metavar='G',
                    help='Value function learning rate (default: 1e-3)')
    parser.add_argument('--clip-param', type=float, default=0.3, metavar='G',
                    help='PPO clip parameter (default: 0.3)')
    parser.add_argument('--ppo-epoch', type=int, default=10, metavar='G',
                    help='Number of PPO epochs (default: 10)')
    parser.add_argument('--num-mini-batch', type=int, default=64, metavar='G',
                    help='Number of PPO mini-batches (default: 64)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, metavar='G',
                    help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, metavar='G',
                    help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, metavar='G',
                    help='Max gradient norm (default: 0.5)')
    parser.add_argument('--use-clipped-value-loss', action='store_true', default=True,
                    help='Use clipped value loss (default: True)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='Batch size for PPO updates (default: 2048)')
    parser.add_argument('--num-steps', type=int, default=10_000_000, metavar='N',
                    help='Maximum number of steps (default: 10_000_000)')
    parser.add_argument('--hidden-size', type=int, default=128, metavar='N',
                    help='Hidden size (default: 128)')
    parser.add_argument('--update-interval', type=int, default=2048, metavar='N',
                    help='Steps between PPO updates (default: 2048)')
    parser.add_argument('--target-update-interval', type=int, default=10, metavar='N',
                    help='Value target update interval (default: 10)')
    parser.add_argument('--cuda', action="store_true", default=True,
                    help='Run on CUDA (default: True)')
    parser.add_argument('--save-curve', action="store_true", default=True,
                    help='Save learning curve plot (default: True)')
    parser.add_argument('--curve-name', type=str, default='ppo_clip_learning_curve',
                    help='Filename for learning curve plot (default: ppo_clip_learning_curve)')
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
    parser.add_argument('--num-envs', type=int, default=8,
                    help='Number of parallel environments (default: 8)')
    parser.add_argument('--render', action="store_true",
                    help='Render visualization (only for training with num-envs=1) (default: False)')

    return parser.parse_args()

def main():
    """Main training function for PPO with vectorized environments and progress bar"""
    # Parse arguments
    args = parse_arguments()
    
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
    print(f"Creating {args.num_envs} parallel environments for '{args.env_name}'")
    
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
    
    # Initialize PPO-CLIP agent
    agent = PPOCLIP(state_dim, action_space, args)
    
    # Set up TensorBoard writer with reduced flush frequency to minimize I/O
    log_dir = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{args.env_name}_ppo_clip_vec{args.num_envs}'
    writer = SummaryWriter(log_dir, max_queue=1000, flush_secs=120)
    
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
    
    # Lists to store metrics for learning curve
    all_episode_rewards = []
    all_episode_numbers = []
    all_eval_rewards = []
    all_eval_episode_numbers = []
    
    # Initialize progress bar for total training
    progress_bar = tqdm(total=args.num_steps, desc="Training Progress", unit="steps")
    
    # Reset environments to get initial states
    states, _ = vec_env.reset(seed=args.seed)
    
    try:
        while total_numsteps < args.num_steps:
            # Run garbage collection periodically
            if updates % args.gc_interval == 0 and updates > 0:
                collect_garbage()
            
            # Start a new rollout collection cycle
            rollouts.clear()
            
            # Collect rollout data
            for step in range(0, steps_per_update):
                # Select actions
                actions, log_probs, values = agent.select_actions_vec(states, evaluate=False)
                
                # Ensure actions have the right shape for the environment
                if isinstance(vec_env.single_action_space, gym.spaces.Discrete):
                    # Convert from array form [n] to scalar form n
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
                        writer.add_scalar('reward/train', episode_rewards[i], episode_count)
                        
                        # Store training metrics for learning curve
                        all_episode_rewards.append(episode_rewards[i])
                        all_episode_numbers.append(episode_count)
                        
                        # Calculate average reward per step
                        reward_per_step = episode_rewards[i] / episode_steps[i] if episode_steps[i] > 0 else 0
                        
                        # Print progress
                        tqdm.write(f"Episode {episode_count}: Reward={episode_rewards[i]:.2f}, Steps={episode_steps[i]}, Reward/Step={reward_per_step:.4f}")
                        
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
            
            # Get final value estimates for computing advantages
            with torch.no_grad():
                final_values = agent.get_value(states)
                
            # Compute returns and advantages using GAE
            rollouts.compute_returns(final_values, args.gamma, args.tau)
            
            # Perform PPO update
            value_loss, policy_loss, entropy_loss, clip_fraction = agent.update_parameters(rollouts)
            
            # Log update statistics
            updates += 1
            if updates % args.log_interval == 0:
                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy', entropy_loss, updates)
                writer.add_scalar('stats/clip_fraction', clip_fraction, updates)
                
                # Print stats
                tqdm.write(f"Update {updates}: Policy Loss={policy_loss:.5f}, Value Loss={value_loss:.5f}, "
                          f"Entropy={entropy_loss:.5f}, Clip Fraction={clip_fraction:.3f}")
            
            # Run evaluation periodically
            if updates % args.eval_interval == 0:
                # Set evaluation message
                progress_bar.set_description("Evaluating...")
                
                # Run evaluation
                eval_mean_reward, eval_std_reward = evaluate_policy_vec(
                    agent, eval_env, max_eval_episodes=args.eval_episodes)
                
                # Log evaluation metrics
                writer.add_scalar('reward/eval', eval_mean_reward, updates)
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
            if updates % args.checkpoint_interval == 0:
                agent.save_checkpoint(args.env_name, suffix=f"update_{updates}")
                # Also save rollouts occasionally for potential resume
                if updates % (args.checkpoint_interval * 5) == 0:
                    rollouts.save_rollouts(args.env_name, suffix=f"update_{updates}")
    
    except KeyboardInterrupt:
        progress_bar.write("\nTraining interrupted by user")
    finally:
        # Close progress bar
        progress_bar.close()
        
        # Final save
        if 'updates' in locals() and updates > 0:
            print(f"Saving final checkpoint at update {updates}")
            agent.save_checkpoint(args.env_name, suffix=f"final_{updates}")
        
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
            
            save_learning_curve(episode_numbers_np, episode_rewards_np, 
                               eval_episode_numbers_np, eval_rewards_np, args.curve_name, args.algorithm)
        
        # Final cleanup
        collect_garbage()
        print("Training finished")

if __name__ == "__main__":
    # Set lower memory usage for numpy operations
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    
    # Set up environment variables to limit CPU utilization if needed
    # os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    # os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads
    
    # Set CUDA options to optimize memory usage
    if torch.cuda.is_available():
        # Try to ensure PyTorch releases memory after operations
        torch.cuda.empty_cache()
        
    # Run main function
    main()