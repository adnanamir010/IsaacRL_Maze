import random

import numpy as np

import os

import pickle
<<<<<<< HEAD


=======
>>>>>>> alternate_timeline

class ReplayMemory:

    """

    Memory-optimized replay buffer for off-policy reinforcement learning algorithms like SAC.

    Uses NumPy arrays for efficient storage and access of transitions.

    """
<<<<<<< HEAD

    def __init__(self, capacity, state_dim, action_dim, seed=0):

=======
    def __init__(self, capacity, state_dim, action_dim, seed=0):
>>>>>>> alternate_timeline
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
<<<<<<< HEAD

    

    def push_batch(self, states, actions, rewards, next_states, dones):

        """

        Add a batch of transitions to the buffer (vectorized version).

=======
    
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
>>>>>>> alternate_timeline
        

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

        
<<<<<<< HEAD

    def push(self, state, action, reward, value, log_prob, mask, entropy=None):

=======
    def push(self, state, action, reward, value, log_prob, mask, action_probs=None, means=None, log_stds=None):
>>>>>>> alternate_timeline
        """

        Add a new transition to the storage.

        

        Args:

            state: Current state

            action: Action taken

            reward: Reward received

            value: Value estimate

            log_prob: Log probability of action

            mask: 1 - done (0 if episode terminated, 1 otherwise)
<<<<<<< HEAD

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

            
=======
            action_probs: Action probabilities for discrete actions
            means: Action means for continuous actions
            log_stds: Action log standard deviations for continuous actions
        """            
        # Ensure we have valid data
        if state is None or np.size(state) == 0:
            print("WARNING: Attempted to push None or empty state!")
            return
            
        if self.position >= self.capacity:
            print(f"WARNING: Position {self.position} exceeds capacity {self.capacity}!")
            return
        
        try:
            # Store the data - use numpy's direct assignment
            self.states[self.position] = state
            self.actions[self.position] = action
            
            # For scalar values, ensure proper conversion
            if isinstance(reward, (list, tuple)) and len(reward) == 1:
                self.rewards[self.position] = reward[0]
            else:
                self.rewards[self.position] = reward
                
            if isinstance(value, (list, tuple)) and len(value) == 1:
                self.values[self.position] = value[0]
            else:
                self.values[self.position] = value
                
            if isinstance(log_prob, (list, tuple)) and len(log_prob) == 1:
                self.log_probs[self.position] = log_prob[0]
            else:
                self.log_probs[self.position] = log_prob
                
            if isinstance(mask, (list, tuple)) and len(mask) == 1:
                self.masks[self.position] = mask[0]
            else:
                self.masks[self.position] = mask
            
            # Store policy distribution parameters if provided
            if action_probs is not None:
                if not hasattr(self, 'action_probs'):
                    # Initialize on first use
                    self.action_probs = np.zeros((self.capacity, len(action_probs)), dtype=np.float32)
                self.action_probs[self.position] = action_probs
                
            if means is not None and log_stds is not None:
                if not hasattr(self, 'means'):
                    # Initialize on first use
                    self.means = np.zeros((self.capacity, len(means)), dtype=np.float32)
                    self.log_stds = np.zeros((self.capacity, len(log_stds)), dtype=np.float32)
                self.means[self.position] = means
                self.log_stds[self.position] = log_stds
            
            # Critical: Increment position
            self.position += 1
            
            # Check if storage is full
            if self.position >= self.capacity:
                self.full = True
                
        except Exception as e:
            print(f"ERROR in push to rollout storage: {e}")
            import traceback
            traceback.print_exc()
>>>>>>> alternate_timeline

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
<<<<<<< HEAD

=======
        
        if num_steps == 0:
            print("Warning: No steps in rollout storage. Cannot compute returns.")
            return
>>>>>>> alternate_timeline
            

        # Initialize advantage
<<<<<<< HEAD

        self.advantages.fill(0)

        last_advantage = 0

        

        # Ensure next_value is properly formatted

        if isinstance(next_value, np.ndarray):

            if next_value.size == 1:

                next_val = float(next_value)  # Convert single-element array to scalar

            else:

                next_val = next_value.flatten()[0]  # Take the first element if multiple

        elif isinstance(next_value, list):

            next_val = next_value[0]  # Take first element if it's a list

        elif hasattr(next_value, 'item'):  # Torch tensor

            next_val = next_value.item()  # Convert tensor to scalar

        else:

            next_val = next_value  # Already a scalar

=======
        self.advantages[:num_steps] = 0.0
        last_advantage = 0.0
        
        # Convert next_value to a scalar if it's not already
        if isinstance(next_value, np.ndarray):
            if next_value.size == 1:
                next_val = float(next_value[0])  # Convert single-element array to scalar
            else:
                next_val = float(next_value.mean())  # Take the mean if multiple values
        elif isinstance(next_value, list):
            next_val = float(np.mean(next_value))  # Take mean if it's a list
        elif hasattr(next_value, 'item'):  # Torch tensor
            next_val = float(next_value.item())  # Convert tensor to scalar
        else:
            next_val = float(next_value)  # Already a scalar
                
        # Compute returns and advantages in reverse order
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
            
        data = {
            'states': self.states[:num_steps],
            'actions': self.actions[:num_steps],
            'rewards': self.rewards[:num_steps],
            'values': self.values[:num_steps],
            'returns': self.returns[:num_steps],
            'advantages': self.advantages[:num_steps],
            'log_probs': self.log_probs[:num_steps],
            'masks': self.masks[:num_steps],
        }
        
        # Add policy distribution parameters if they exist
        if hasattr(self, 'action_probs'):
            data['action_probs'] = self.action_probs[:num_steps]
        if hasattr(self, 'means') and hasattr(self, 'log_stds'):
            data['means'] = self.means[:num_steps]
            data['log_stds'] = self.log_stds[:num_steps]
            
        return data

    def debug_info(self):
        """Print debug information about storage state"""
        position = self.position
        capacity = self.capacity
        print(f"RolloutStorage Debug - Position: {position}/{capacity}")
        
        # Check if data exists
        has_states = hasattr(self, 'states') and self.states is not None
        has_actions = hasattr(self, 'actions') and self.actions is not None
        
        print(f"Storage has states: {has_states}, has actions: {has_actions}")
        
        if has_states and position > 0:
            print(f"First state sample: {self.states[0][:5]}...")
            print(f"Last state sample: {self.states[position-1][:5]}...")
        
        return position > 0

    def clear(self):
        """Reset the storage by simply resetting position."""
        self.position = 0
        self.full = False
>>>>>>> alternate_timeline
        

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

            if isinstance(last_advantage, (np.ndarray, list)) and len(last_advantage) == 1:

                last_advantage = float(last_advantage[0])  # Convert to scalar if it's a single-element array

            

            last_advantage = delta + gamma * gae_lambda * self.masks[step] * last_advantage

            

            # Handle the case where last_advantage is still an array

            if isinstance(last_advantage, (np.ndarray, list)):

                if len(last_advantage) == 1:

                    self.advantages[step] = float(last_advantage[0])

                else:

                    # Just take the first element as a fallback

                    self.advantages[step] = float(last_advantage[0])

            else:

                # It's already a scalar

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