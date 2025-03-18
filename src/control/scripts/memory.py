import random
import numpy as np
import os
import pickle
from collections import deque, namedtuple

class ReplayMemory:
    """
    Replay buffer for off-policy reinforcement learning algorithms like SAC.
    Stores transitions (state, action, reward, next_state, done) for random sampling.
    """
    def __init__(self, capacity, seed=0):
        """
        Initialize replay buffer.
        
        Args:
            capacity (int): Maximum size of the buffer
            seed (int): Random seed for reproducibility
        """
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def save_buffer(self, env_name, suffix="", save_path=None):
        """
        Save the replay buffer to a file.
        
        Args:
            env_name (str): Environment name
            suffix (str): Additional suffix for filename
            save_path (str): Path to save buffer, if None uses default
        """
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_path is None:
            save_path = f"checkpoints/sac_buffer_{env_name}_{suffix}"
        print(f'Saving buffer to {save_path}')
        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
            
    def load_buffer(self, save_path):
        """
        Load the replay buffer from a file.
        
        Args:
            save_path (str): Path to load buffer from
        """
        print(f'Loading buffer from {save_path}')
        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.capacity


class RolloutStorage:
    """
    Storage for on-policy reinforcement learning algorithms like PPO.
    Collects full trajectories and computes returns and advantages.
    """
    def __init__(self, capacity, state_dim, action_dim, seed=0):
        """
        Initialize rollout storage.
        
        Args:
            capacity (int): Maximum number of time steps to store
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            seed (int): Random seed for reproducibility
        """
        random.seed(seed)
        self.capacity = capacity
        
        # Initialize storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.masks = np.ones(capacity, dtype=np.float32)  # 1 for not done, 0 for done
        
        self.entropies = np.zeros(capacity, dtype=np.float32)  # Optional: store entropies
        
        self.position = 0
        self.full = False
        
    def push(self, state, action, reward, value, log_prob, mask, entropy=None):
        """
        Add a new transition to the storage.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            mask: 1 - done (0 if episode terminated, 1 otherwise)
            entropy: Action distribution entropy (optional)
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.masks[self.position] = mask
        
        if entropy is not None:
            self.entropies[self.position] = entropy
            
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True
            
    def compute_returns(self, next_value, gamma, gae_lambda):
        """
        Compute returns and advantages using Generalized Advantage Estimation.
        
        Args:
            next_value: Value estimate for the state after the last stored state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Calculate how many steps we actually have
        if self.full:
            num_steps = self.capacity
        else:
            num_steps = self.position
            
        # Initialize advantage
        advantage = 0
        
        # Compute returns and advantages in reverse order
        for step in reversed(range(num_steps)):
            # For the last step, use the provided next_value
            if step == num_steps - 1:
                next_value_step = next_value
            else:
                next_value_step = self.values[step + 1]
                
            # Calculate delta (TD error)
            delta = self.rewards[step] + gamma * next_value_step * self.masks[step] - self.values[step]
            
            # Update advantage using GAE
            advantage = delta + gamma * gae_lambda * self.masks[step] * advantage
            
            # Store return (advantage + value)
            self.returns[step] = advantage + self.values[step]
            
    def get_data(self):
        """
        Get all stored data as a dictionary.
        
        Returns:
            dict: Dictionary containing all stored data
        """
        if self.full:
            num_steps = self.capacity
        else:
            num_steps = self.position
            
        return {
            'states': self.states[:num_steps],
            'actions': self.actions[:num_steps],
            'rewards': self.rewards[:num_steps],
            'values': self.values[:num_steps],
            'returns': self.returns[:num_steps],
            'log_probs': self.log_probs[:num_steps],
            'masks': self.masks[:num_steps],
            'entropies': self.entropies[:num_steps]
        }
    
    def clear(self):
        """Reset the storage."""
        self.position = 0
        self.full = False
        
    def __len__(self):
        """Return the current size of the storage."""
        if self.full:
            return self.capacity
        return self.position
    
    def save_rollouts(self, env_name, suffix="", save_path=None):
        """
        Save the rollout data to a file.
        
        Args:
            env_name (str): Environment name
            suffix (str): Additional suffix for filename
            save_path (str): Path to save rollouts, if None uses default
        """
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if save_path is None:
            save_path = f"checkpoints/ppo_rollouts_{env_name}_{suffix}"
        print(f'Saving rollouts to {save_path}')
        
        data = self.get_data()
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_rollouts(self, save_path):
        """
        Load rollout data from a file.
        
        Args:
            save_path (str): Path to load rollouts from
        """
        print(f'Loading rollouts from {save_path}')
        with open(save_path, "rb") as f:
            data = pickle.load(f)
            
        # Fill the storage with loaded data
        num_steps = len(data['states'])
        self.states[:num_steps] = data['states']
        self.actions[:num_steps] = data['actions']
        self.rewards[:num_steps] = data['rewards']
        self.values[:num_steps] = data['values']
        self.returns[:num_steps] = data['returns']
        self.log_probs[:num_steps] = data['log_probs']
        self.masks[:num_steps] = data['masks']
        self.entropies[:num_steps] = data['entropies']
        
        self.position = num_steps % self.capacity
        self.full = (num_steps >= self.capacity)