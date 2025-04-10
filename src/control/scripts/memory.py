import random
import numpy as np
import os
import pickle

class ReplayMemory:
    """
    Memory-optimized replay buffer for off-policy reinforcement learning algorithms like SAC.
    Uses NumPy arrays for efficient storage and access of transitions.
    """
    def __init__(self, capacity, state_dim, action_dim, seed=0):
        """
        Initialize optimized replay buffer with pre-allocated NumPy arrays.
        
        Args:
            capacity (int): Maximum size of the buffer
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            seed (int): Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pre-allocate fixed NumPy arrays for memory efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # Using float for mask values
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated (or mask value)
        """
        # Store transition in pre-allocated arrays
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        Add a batch of transitions to the buffer (vectorized version).
        
        Args:
            states: Batch of current states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of episode termination flags
        """
        batch_size = len(states)
        
        # Calculate indices for the batch, handling buffer wrapping
        indices = np.arange(self.position, self.position + batch_size) % self.capacity
        
        # Store transitions
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones
        
        # Update position and size
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions using efficient NumPy operations.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        # Efficiently sample indices without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Return views into the arrays to avoid unnecessary copying
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size
    
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
        
        # Create a dictionary with buffer data and metadata
        buffer_dict = {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_states': self.next_states[:self.size],
            'dones': self.dones[:self.size],
            'position': self.position,
            'size': self.size,
            'capacity': self.capacity
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(buffer_dict, f, protocol=4)  # Using protocol 4 for better efficiency
            
    def load_buffer(self, save_path):
        """
        Load the replay buffer from a file.
        
        Args:
            save_path (str): Path to load buffer from
        """
        print(f'Loading buffer from {save_path}')
        with open(save_path, "rb") as f:
            buffer_dict = pickle.load(f)
        
        # Load data and metadata
        data_size = buffer_dict['size']
        
        # Resize arrays if capacity doesn't match
        if self.capacity < data_size:
            self.capacity = data_size
            self.states = np.zeros((self.capacity, self.states.shape[1]), dtype=np.float32)
            self.actions = np.zeros((self.capacity, self.actions.shape[1]), dtype=np.float32)
            self.rewards = np.zeros(self.capacity, dtype=np.float32)
            self.next_states = np.zeros((self.capacity, self.next_states.shape[1]), dtype=np.float32)
            self.dones = np.zeros(self.capacity, dtype=np.float32)
        
        # Copy data to buffer arrays
        self.states[:data_size] = buffer_dict['states']
        self.actions[:data_size] = buffer_dict['actions']
        self.rewards[:data_size] = buffer_dict['rewards']
        self.next_states[:data_size] = buffer_dict['next_states']
        self.dones[:data_size] = buffer_dict['dones']
        
        self.position = buffer_dict['position']
        self.size = data_size


class RolloutStorage:
    """
    Memory-optimized storage for on-policy reinforcement learning algorithms like PPO.
    Uses NumPy arrays for efficient storage and computation of returns and advantages.
    """
    def __init__(self, capacity, state_dim, action_dim, seed=0):
        """
        Initialize rollout storage with pre-allocated NumPy arrays.
        
        Args:
            capacity (int): Maximum number of time steps to store
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            seed (int): Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize storage with pre-allocated NumPy arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.masks = np.ones(capacity, dtype=np.float32)  # 1 for not done, 0 for done
        self.advantages = np.zeros(capacity, dtype=np.float32)  # Pre-allocate advantages
        self.entropies = np.zeros(capacity, dtype=np.float32)  # Store entropies
        
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
        Uses vectorized NumPy operations for efficiency.
        
        Args:
            next_value: Value estimate for the state after the last stored state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Calculate how many steps we actually have
        num_steps = self.capacity if self.full else self.position
            
        # Initialize advantage
        self.advantages.fill(0)
        last_advantage = 0
        
        # Ensure next_value is flattened to match expected shape
        if hasattr(next_value, 'shape') and len(next_value.shape) > 1:
            next_val = next_value.flatten()
        else:
            next_val = next_value
        
        # Compute returns and advantages in reverse order (more efficient in NumPy)
        for step in reversed(range(num_steps)):
            # For the last step, use the provided next_value
            if step == num_steps - 1:
                next_val_step = next_val
            else:
                next_val_step = self.values[step + 1]
                
            # Calculate delta (TD error)
            delta = self.rewards[step] + gamma * next_val_step * self.masks[step] - self.values[step]
            
            # Update advantage using GAE
            last_advantage = delta + gamma * gae_lambda * self.masks[step] * last_advantage
            
            # Make sure last_advantage is a scalar if it's a single-element array
            if hasattr(last_advantage, 'shape') and last_advantage.size == 1:
                last_advantage = float(last_advantage)
                
            self.advantages[step] = last_advantage
            
        # Calculate returns (advantage + value)
        self.returns[:num_steps] = self.advantages[:num_steps] + self.values[:num_steps]

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
            
        # Return slices of the arrays to avoid copying
        return {
            'states': self.states[:num_steps],
            'actions': self.actions[:num_steps],
            'rewards': self.rewards[:num_steps],
            'values': self.values[:num_steps],
            'returns': self.returns[:num_steps],
            'advantages': self.advantages[:num_steps],  # Include pre-computed advantages
            'log_probs': self.log_probs[:num_steps],
            'masks': self.masks[:num_steps],
            'entropies': self.entropies[:num_steps]
        }
    
    def clear(self):
        """Reset the storage by simply resetting position."""
        self.position = 0
        self.full = False
        
    def __len__(self):
        """Return the current size of the storage."""
        if self.full:
            return self.capacity
        return self.position
    
    def save_rollouts(self, env_name, suffix="", save_path=None):
        """
        Save the rollout data to a file using optimal pickle protocol.
        
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
            pickle.dump(data, f, protocol=4)  # Using protocol 4 for better efficiency
            
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
        
        # Resize arrays if necessary
        if num_steps > self.capacity:
            self.capacity = num_steps
            self._resize_arrays()
        
        # Copy data to arrays
        self.states[:num_steps] = data['states']
        self.actions[:num_steps] = data['actions']
        self.rewards[:num_steps] = data['rewards']
        self.values[:num_steps] = data['values']
        self.returns[:num_steps] = data['returns']
        self.log_probs[:num_steps] = data['log_probs']
        self.masks[:num_steps] = data['masks']
        self.entropies[:num_steps] = data['entropies']
        
        # Load advantages if available
        if 'advantages' in data:
            self.advantages[:num_steps] = data['advantages']
        
        self.position = num_steps % self.capacity
        self.full = (num_steps >= self.capacity)
        
    def _resize_arrays(self):
        """Resize internal arrays if capacity changes."""
        self.states = np.resize(self.states, (self.capacity, self.state_dim))
        self.actions = np.resize(self.actions, (self.capacity, self.action_dim))
        self.rewards = np.resize(self.rewards, self.capacity)
        self.values = np.resize(self.values, self.capacity)
        self.returns = np.resize(self.returns, self.capacity)
        self.log_probs = np.resize(self.log_probs, self.capacity)
        self.masks = np.resize(self.masks, self.capacity)
        self.advantages = np.resize(self.advantages, self.capacity)
        self.entropies = np.resize(self.entropies, self.capacity)