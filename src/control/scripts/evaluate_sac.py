#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import gymnasium as gym
import time
from agents import SAC
from rl_utils import evaluate_policy
from environment import make_vec_env

def parse_arguments():
    """Parse command line arguments for the evaluation script"""
    parser = argparse.ArgumentParser(description='Evaluate a trained SAC agent')
    parser.add_argument('--env-name', default="gym_navigation:NavigationGoal-v0",
                    help='Environment name (default: gym_navigation:NavigationGoal-v0)')
    parser.add_argument('--track-id', type=int, default=0,
                    help='Track ID for NavigationGoal environment (default: 0)')
    parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to the checkpoint file')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='Random seed (default: 123456)')
    parser.add_argument('--num-episodes', type=int, default=10,
                    help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--render', action='store_true', default=False,
                    help='Render the environment (default: False)')
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
    
    if args.env_name == "gym_navigation:NavigationGoal-v0":
        env = gym.make(args.env_name, render_mode=render_mode, track_id=args.track_id)
    else:
        try:
            env = gym.make(args.env_name, render_mode=render_mode)
        except TypeError:
            # Fallback for environments that don't support render_mode
            env = gym.make(args.env_name)
    
    # Set environment seed
    env.seed(args.seed)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create action space for SAC
    class ActionSpace:
        def __init__(self, low, high, shape):
            self.low = low
            self.high = high
            self.shape = shape
    
    action_space = ActionSpace(
        env.action_space.low,
        env.action_space.high,
        env.action_space.shape
    )
    
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
    agent = SAC(state_dim, action_space, agent_args)
    
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
            action = agent.select_action(state, evaluate=True)
            
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