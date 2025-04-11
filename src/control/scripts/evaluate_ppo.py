#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import gymnasium as gym
import time
from agents import PPOCLIP, PPOKL
from rl_utils import evaluate_policy
from environment import VectorizedDDEnv, make_vectorized_env

def parse_arguments():
    """Parse command line arguments for the evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO agent')
    parser.add_argument('--env-name', default="VectorizedDD",
                    help='Environment name (default: VectorizedDD)')
    parser.add_argument('--obstacle-shape', default="square",
                    help='Obstacle shape: circular | square (default: square)')
    parser.add_argument('--algorithm', default="PPOCLIP", choices=["PPOCLIP", "PPOKL"],
                    help='Algorithm: PPOCLIP | PPOKL (default: PPOCLIP)')
    parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to the checkpoint file')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--num-episodes', type=int, default=10,
                    help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--render', action='store_true', default=True,
                    help='Render the environment (default: False)')
    parser.add_argument('--hidden-size', type=int, default=512,
                    help='Hidden layer size (default: 128)')
    parser.add_argument('--cuda', action="store_true", default=True,
                    help='Run on CUDA if available (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='Discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='GAE parameter (default: 0.95)')
    parser.add_argument('--policy-lr', type=float, default=3e-4, metavar='G',
                    help='Policy learning rate (default: 3e-4)')
    parser.add_argument('--value-lr', type=float, default=1e-3, metavar='G',
                    help='Value function learning rate (default: 1e-3)')

    # PPO-CLIP specific arguments
    parser.add_argument('--clip-param', type=float, default=0.2,
                    help='PPO clip parameter (default: 0.2)')
    parser.add_argument('--ppo-epoch', type=int, default=10,
                    help='Number of PPO epochs (default: 10)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                    help='Number of PPO mini-batches (default: 32)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='Max gradient norm (default: 0.5)')
    parser.add_argument('--use-clipped-value-loss', action='store_true', default=True,
                    help='Use clipped value loss (default: True)')
                    
    # PPO-KL specific arguments
    parser.add_argument('--kl-target', type=float, default=0.01,
                    help='Target KL divergence (default: 0.01)')
    parser.add_argument('--kl-coef', type=float, default=0.2,
                    help='Initial KL coefficient (default: 0.2)')
    parser.add_argument('--kl-adaptive', action='store_true', default=True,
                    help='Use adaptive KL coefficient (default: True)')
    parser.add_argument('--kl-cutoff-factor', type=float, default=2.0,
                    help='KL cutoff factor (default: 2.0)')
    parser.add_argument('--kl-cutoff-coef', type=float, default=1000.0,
                    help='KL cutoff coefficient (default: 1000.0)')

    return parser.parse_args()

def main():
    """Evaluate a trained PPO agent"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure device
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    render_mode = 'human' if args.render else None
    
    if args.env_name == "VectorizedDD":
        env = VectorizedDDEnv(render_mode=render_mode, obstacle_shape=args.obstacle_shape)
    else:
        try:
            env = gym.make(args.env_name, render_mode=render_mode)
        except TypeError:
            # Fallback for environments that don't support render_mode
            env = gym.make(args.env_name)
    
    # Set environment seed
    try:
        env.reset(seed=args.seed)
    except (TypeError, AttributeError):
        try:
            env.seed(args.seed)
        except (AttributeError, TypeError):
            pass  # Ignore if seeding is not supported
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    
    # Handle different types of action spaces
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        print(f"Continuous action space detected with {action_dim} dimensions")
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = 1  # Discrete action is represented as a single integer
        print(f"Discrete action space detected with {env.action_space.n} possible actions")
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")
        
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Initialize agent
    if args.algorithm == "PPOCLIP":
        agent = PPOCLIP(state_dim, env.action_space, args)
    else:  # PPOKL
        agent = PPOKL(state_dim, env.action_space, args)
    
    # Load checkpoint
    agent.load_checkpoint(args.checkpoint, evaluate=True)
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Evaluate agent
    total_reward = 0
    total_steps = 0
    
    for episode in range(args.num_episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Handle gymnasium reset which returns (obs, info)
        try:
            result = env.reset(seed=args.seed + episode)
            if isinstance(result, tuple):
                state, _ = result  # Gymnasium reset
            else:
                state = result     # Old gym reset
        except TypeError:
            # Fallback for environments with different reset signatures
            state = env.reset()
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(state, evaluate=True)
            
            # For discrete actions, ensure correct format
            if isinstance(env.action_space, gym.spaces.Discrete):
                if isinstance(action, np.ndarray) and action.size == 1:
                    action = action[0]
                if hasattr(action, 'item'):
                    action = action.item()
            
            # Take step in environment
            result = env.step(action)
            
            # Handle different step return formats
            if len(result) == 5:  # Gymnasium format: obs, reward, terminated, truncated, info
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:  # Old gym format: obs, reward, done, info
                next_state, reward, done, _ = result
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Render if requested (already handled by render_mode)
            if args.render and render_mode != "human":
                env.render()
                time.sleep(0.01)  # Small delay to make rendering visible
        
        # Update totals
        total_reward += episode_reward
        total_steps += episode_steps
        
        # Print episode stats
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={episode_steps}")
    
    # Print overall stats
    print(f"\nEvaluation over {args.num_episodes} episodes:")
    print(f"Mean Reward: {total_reward/args.num_episodes:.2f}")
    print(f"Mean Steps: {total_steps/args.num_episodes:.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()