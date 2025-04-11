import math
import torch
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import gc
import gymnasium as gym

def soft_update(target, source, tau):
    """
    Perform soft update of target network parameters from source network.
    Memory-optimized implementation.
    
    Args:
        target: Target network
        source: Source network
        tau: Update rate (0 < tau <= 1)
    """
    with torch.no_grad():  # Avoid storing gradients to save memory
        for target_param, param in zip(target.parameters(), source.parameters()):
            # Inplace operations to save memory
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(param.data, alpha=tau)

def hard_update(target, source):
    """
    Copy parameters from source network to target network.
    Memory-optimized implementation.
    
    Args:
        target: Target network
        source: Source network
    """
    with torch.no_grad():  # Avoid storing gradients
        for target_param, param in zip(target.parameters(), source.parameters()):
            # Direct copy without creating intermediate tensors
            target_param.data.copy_(param.data)

def evaluate_policy(agent, env, max_eval_episodes=10, max_steps=1000, seed=None):
    """
    Evaluate the agent's policy in the environment.
    
    Args:
        agent: Agent to evaluate
        env: Environment to evaluate in
        max_eval_episodes: Maximum number of episodes to evaluate
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        
    Returns:
        float: Mean episode reward
        float: Standard deviation of episode rewards
    """
    # Set evaluation seed if provided
    if seed is not None:
        try:
            env.reset(seed=seed)
        except (TypeError, ValueError):
            try:
                env.seed(seed)
            except (AttributeError, TypeError):
                pass  # Ignore if seeding is not supported
    
    # Use no_grad mode to save memory
    with torch.no_grad():
        episode_rewards = []
        
        for episode in range(max_eval_episodes):
            # Reset environment
            try:
                result = env.reset()
                if isinstance(result, tuple):
                    state, _ = result  # Gymnasium reset
                else:
                    state = result  # Old gym reset
            except TypeError:
                # Fallback for environments with different reset signatures
                state = env.reset()
            
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Select action deterministically
                action = agent.select_action(state, evaluate=True)
                
                # For discrete action spaces, convert the action to an integer
                if isinstance(env.action_space, gym.spaces.Discrete):
                    if isinstance(action, np.ndarray):
                        action = action.item() if action.size == 1 else int(action)
                
                # Take step in environment
                result = env.step(action)
                
                # Handle different step return formats
                if len(result) == 5:  # Gymnasium format: obs, reward, terminated, truncated, info
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:  # Old gym format: obs, reward, done, info
                    next_state, reward, done, _ = result
                
                episode_reward += reward
                state = next_state
                step += 1
            
            episode_rewards.append(episode_reward)
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
    
    return mean_reward, std_reward

def evaluate_policy_vec(agent, vec_env, max_eval_episodes=10, max_steps=1000, seed=None):
    """
    Evaluate the agent's policy in vectorized environments for faster evaluation.
    Fixed version that handles potential empty state arrays.
    
    Args:
        agent: Agent to evaluate
        vec_env: Vectorized environment
        max_eval_episodes: Maximum number of episodes to evaluate
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        
    Returns:
        float: Mean episode reward
        float: Standard deviation of episode rewards
    """
    # Set evaluation seed if provided
    if seed is not None:
        try:
            vec_env.seed(seed)
        except (AttributeError, TypeError):
            pass  # Ignore if seeding is not supported
    
    # Use torch.no_grad() to avoid storing gradients during evaluation
    with torch.no_grad():
        num_envs = vec_env.num_envs
        episode_rewards = []
        
        # Reset all environments
        states, _ = vec_env.reset(seed=seed)
        
        # Track which environments have finished their episodes
        env_dones = [False] * num_envs
        env_rewards = [0] * num_envs
        env_steps = [0] * num_envs
        
        # Run until we collect max_eval_episodes episodes
        while len(episode_rewards) < max_eval_episodes:
            # Select actions for all active environments
            if any(not done for done in env_dones):  # Only process if any environment is active
                # Create a mask for active environments
                active_env_indices = [i for i, done in enumerate(env_dones) if not done]
                active_states = states[active_env_indices]
                
                if len(active_states) > 0:  # Ensure we have states to process
                    # Get actions only for active environments
                    all_actions = np.zeros((num_envs,) + vec_env.single_action_space.shape, dtype=np.float32)
                    active_actions = agent.select_actions_vec(active_states, evaluate=True)
                    
                    # Place active actions in the correct indices
                    for idx, action in zip(active_env_indices, active_actions):
                        all_actions[idx] = action
                    
                    # Step all environments
                    next_states, rewards, terminations, truncations, infos = vec_env.step(all_actions)
                    dones = np.logical_or(terminations, truncations)
                    
                    # Update states and steps
                    for env_idx in range(num_envs):
                        # Skip environments that have already completed their episodes
                        if env_dones[env_idx]:
                            continue
                        
                        # Update rewards and steps
                        env_rewards[env_idx] += rewards[env_idx]
                        env_steps[env_idx] += 1
                        
                        # Check for episode termination
                        if dones[env_idx] or env_steps[env_idx] >= max_steps:
                            episode_rewards.append(env_rewards[env_idx])
                            env_dones[env_idx] = True
                            
                            # Reset this environment if we need more episodes
                            if len(episode_rewards) < max_eval_episodes:
                                # Reset just this environment's tracking variables
                                env_rewards[env_idx] = 0
                                env_steps[env_idx] = 0
                                env_dones[env_idx] = False
                    
                    # Update states for next iteration
                    states = next_states
            
            # If all environments are done, reset all
            if all(env_dones) and len(episode_rewards) < max_eval_episodes:
                states, _ = vec_env.reset(seed=seed)
                env_dones = [False] * num_envs
                env_rewards = [0] * num_envs
                env_steps = [0] * num_envs
        
        # Truncate to the requested number of episodes
        episode_rewards = episode_rewards[:max_eval_episodes]
        
        # Calculate statistics on the collected episode rewards
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
    return mean_reward, std_reward

def clear_cuda_cache():
    """Clear CUDA cache to free up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def collect_garbage():
    """Force garbage collection"""
    gc.collect()
    clear_cuda_cache()

def save_learning_curve(episodes, rewards, eval_episodes=None, eval_rewards=None, filename='learning_curve', algorithm='SAC'):
    """
    Save a plot of the learning curve showing training and evaluation rewards with confidence bands.
    Memory-optimized implementation.
    
    Args:
        episodes: List of episode numbers for training
        rewards: List of training episode rewards
        eval_episodes: List of episode numbers for evaluation
        eval_rewards: List of evaluation rewards
        filename: Filename for the saved plot
    """
    # Convert lists to numpy arrays for more efficient operations
    episodes = np.array(episodes, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    
    # Create figure with reduced DPI to save memory
    plt.figure(figsize=(10, 6), dpi=80)
    
    # Plot training rewards
    plt.plot(episodes, rewards, 'b-', alpha=0.2, label='Training Rewards')
    
    # Add a smoothed version of training rewards with confidence bands
    window_size = min(len(rewards) // 5, 10) if len(rewards) > 10 else 1
    if window_size > 1:
        # Pre-allocate arrays for smoothed values and confidence bands
        smoothed_rewards = np.zeros_like(rewards)
        confidence_upper = np.zeros_like(rewards)
        confidence_lower = np.zeros_like(rewards)
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000  # Process 1000 points at a time
        for chunk_start in range(0, len(rewards), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(rewards))
            chunk_indices = np.arange(chunk_start, chunk_end)
            
            for i in chunk_indices:
                start_idx = max(0, i - window_size)
                end_idx = min(len(rewards), i + window_size + 1)
                window_rewards = rewards[start_idx:end_idx]
                
                # Calculate mean and standard deviation for the window
                mean_reward = np.mean(window_rewards)
                smoothed_rewards[i] = mean_reward
                
                # Calculate standard deviation for confidence bands
                if len(window_rewards) > 1:
                    std_dev = np.std(window_rewards)
                    # 95% confidence interval
                    confidence_upper[i] = mean_reward + 1.96 * std_dev / np.sqrt(len(window_rewards))
                    confidence_lower[i] = mean_reward - 1.96 * std_dev / np.sqrt(len(window_rewards))
                else:
                    confidence_upper[i] = mean_reward
                    confidence_lower[i] = mean_reward
        
        # Plot smoothed line
        plt.plot(episodes, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed Training (window={window_size*2+1})')
        
        # Plot confidence bands
        plt.fill_between(episodes, confidence_lower, confidence_upper, color='b', alpha=0.2, label='95% Confidence Interval')
    
    # Plot evaluation rewards if available
    if eval_episodes is not None and eval_rewards is not None:
        eval_episodes = np.array(eval_episodes, dtype=np.float32)
        eval_rewards = np.array(eval_rewards, dtype=np.float32)
        
        plt.plot(eval_episodes, eval_rewards, 'r-', linewidth=2, label='Evaluation Rewards')
        
        # If we have multiple evaluation points, add a smoothed trend line
        if len(eval_rewards) > 3:
            eval_smooth = np.zeros_like(eval_rewards)
            for i in range(len(eval_rewards)):
                start_idx = max(0, i - 1)
                end_idx = min(len(eval_rewards), i + 2)
                eval_smooth[i] = np.mean(eval_rewards[start_idx:end_idx])
            plt.plot(eval_episodes, eval_smooth, 'r--', linewidth=1.5, alpha=0.7, label='Evaluation Trend')
    
    # Add labels and title with optimized settings
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'{algorithm} Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use a tight layout to optimize figure size
    plt.tight_layout()
    
    # Save the figure with reduced quality to save disk space
    curves_dir = 'learning_curves'
    if not os.path.exists(curves_dir):
        os.makedirs(curves_dir)
        
    # Add timestamp to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = f"{curves_dir}/{filename}_{timestamp}.png"
    
    plt.savefig(full_path, dpi=80, bbox_inches='tight', pad_inches=0.1)
    print(f"Learning curve saved to {full_path}")
    
    # Release memory
    plt.close()

# Function to create the gym-navigation environment
def make_navigation_env(track_id=0, render_mode=None):
    """
    Create the gym-navigation environment by Nick Geramanis.
    
    Args:
        track_id (int): ID of the track to use (default: 0)
        render_mode (str): Rendering mode, e.g., 'human', 'rgb_array' (default: None)
        
    Returns:
        gym.Env: Navigation environment
    """
    try:
        import gymnasium as gym
        
        # Create the gym-navigation environment
        env = gym.make('gym_navigation:NavigationGoal-v0', 
                       render_mode=render_mode, 
                       track_id=track_id)
        return env
    except ImportError:
        raise ImportError("Please install gymnasium and gym-navigation: "
                         "pip install gymnasium gym-navigation")