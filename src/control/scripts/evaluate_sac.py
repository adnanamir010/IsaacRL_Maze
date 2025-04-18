#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import gymnasium as gym
import time
from agents import SAC
from rl_utils import evaluate_policy
from environment import VectorizedDDEnv, make_vectorized_env

def parse_arguments():
    """Parse command line arguments for the evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate a trained SAC agent')
    parser.add_argument('--env-name', default="VectorizedDD",
                    help='Environment name (default: VectorizedDD)')
    parser.add_argument('--obstacle-shape', default="square",
                    help='Obstacle shape: circular | square (default: square)')
    parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to the checkpoint file')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--num-episodes', type=int, default=10,
                    help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--render', action='store_true', default=True,
                    help='Render the environment (default: True)')
    parser.add_argument('--delay', type=float, default=0.01,
                    help='Delay between rendered frames (default: 0.01s)')
    parser.add_argument('--hidden-size', type=int, default=256,
                    help='Hidden layer size (default: 256)')
    parser.add_argument('--cuda', action="store_true", default=True,
                    help='Run on CUDA if available (default: True)')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

    return parser.parse_args()

def main():
    """Evaluate a trained SAC agent"""
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
    
    # Initialize SAC agent
    class Args:
        def __init__(self):
            self.gamma = 0.99
            self.tau = 0.005
            self.lr = 0.0003
            self.alpha = 0.2
            self.policy = args.policy
            self.target_update_interval = 1
            self.hidden_size = args.hidden_size
            self.automatic_entropy_tuning = True
            self.cuda = use_cuda
    
    agent_args = Args()
    agent = SAC(state_dim, env.action_space, agent_args)
    
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
        
        print(f"Episode {episode+1}: Starting evaluation...")
        
        while not done:
            # Select action
            action = agent.select_action(state, evaluate=True)
            
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
            
            # Render if requested (and not already handled by render_mode)
            if args.render and render_mode != "human":
                env.render()
                time.sleep(args.delay)  # Add delay to make rendering viewable
        
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