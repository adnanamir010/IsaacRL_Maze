import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Constants for numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

def weights_init_(m):
    """
    Initialize neural network weights using Xavier uniform initialization.
    
    Args:
        m: Neural network module
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    """
    Value network that estimates state values for PPO.
    """
    def __init__(self, num_inputs, hidden_dim, init_w=3e-3):
        """
        Initialize Value Network.
        
        Args:
            num_inputs (int): Dimension of state space
            hidden_dim (int): Size of hidden layers
            init_w (float): Weight initialization range
        """
        super(ValueNetwork, self).__init__()
        
        # MLP for value function
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # Weight initialization
        self.apply(weights_init_)
        
    def forward(self, state):
        """
        Forward pass to get state value.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: State value estimate
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x


class QNetwork(nn.Module):
    """
    Critic network for SAC that estimates Q-values (state-action values).
    Implements the double Q-network architecture.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, discrete=False):
        """
        Initialize Q-Network.
        
        Args:
            num_inputs (int): Dimension of the state space
            num_actions (int): Dimension of the action space
            hidden_dim (int): Size of hidden layers
            discrete (bool): Whether the action space is discrete
        """
        super(QNetwork, self).__init__()
        
        self.discrete = discrete
        
        if discrete:
            # For discrete actions, we don't concatenate actions with states
            # Q1 architecture
            self.linear1_q1 = nn.Linear(num_inputs, hidden_dim)
            self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3_q1 = nn.Linear(hidden_dim, num_actions)  # Outputs Q-value for each action
            
            # Q2 architecture (for double Q-learning)
            self.linear1_q2 = nn.Linear(num_inputs, hidden_dim)
            self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3_q2 = nn.Linear(hidden_dim, num_actions)  # Outputs Q-value for each action
        else:
            # Continuous actions - concatenate action with state
            # Q1 architecture
            self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, 1)

            # Q2 architecture (for double Q-learning)
            self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.linear5 = nn.Linear(hidden_dim, hidden_dim)
            self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action=None):
        """
        Compute Q-values for given state-action pairs.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor (only needed for continuous action spaces)
            
        Returns:
            tuple: (Q1 value, Q2 value)
        """
        if self.discrete:
            # For discrete actions, we output Q-values for all actions
            x1 = F.relu(self.linear1_q1(state))
            x1 = F.relu(self.linear2_q1(x1))
            q1 = self.linear3_q1(x1)
            
            x2 = F.relu(self.linear1_q2(state))
            x2 = F.relu(self.linear2_q2(x2))
            q2 = self.linear3_q2(x2)
            
            # If action is provided, use it to select specific Q-values
            if action is not None:
                action_indices = action.long()
                q1 = q1.gather(1, action_indices)
                q2 = q2.gather(1, action_indices)
                
            return q1, q2
        else:
            # Continuous actions - concatenate action with state
            xu = torch.cat([state, action], 1)
            
            # First Q-network
            x1 = F.relu(self.linear1(xu))
            x1 = F.relu(self.linear2(x1))
            x1 = self.linear3(x1)

            # Second Q-network
            x2 = F.relu(self.linear4(xu))
            x2 = F.relu(self.linear5(x2))
            x2 = self.linear6(x2)

            return x1, x2
    
    def Q1(self, state, action=None):
        """
        Returns only the first Q-value, useful for computing actor loss.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor (only needed for continuous action spaces)
            
        Returns:
            torch.Tensor: Q1 value
        """
        if self.discrete:
            # For discrete actions
            x1 = F.relu(self.linear1_q1(state))
            x1 = F.relu(self.linear2_q1(x1))
            q1 = self.linear3_q1(x1)
            
            # If action is provided, select specific Q-values
            if action is not None:
                action_indices = action.long()
                q1 = q1.gather(1, action_indices)
                
            return q1
        else:
            # For continuous actions
            xu = torch.cat([state, action], 1)
            
            x1 = F.relu(self.linear1(xu))
            x1 = F.relu(self.linear2(x1))
            x1 = self.linear3(x1)
            
            return x1


class GaussianPolicy(nn.Module):
    """
    Stochastic policy that outputs a Gaussian distribution over actions.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        """
        Initialize Gaussian Policy.
        
        Args:
            num_inputs (int): Dimension of the state space
            num_actions (int): Dimension of the action space
            hidden_dim (int): Size of hidden layers
            action_space: Action space with high and low bounds (optional)
        """
        super(GaussianPolicy, self).__init__()
        
        # MLP layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Policy head for mean and log standard deviation
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # Action scaling parameters
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """
        Compute action distribution parameters.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (mean, log_std) of action distribution
        """
        # Forward pass through shared layers
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        # Compute mean and log standard deviation
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std

    def sample(self, state):
        """
        Sample actions from the policy.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action, log_prob, mean) - sampled action, log probability, and mean
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        # Scale actions to the correct range
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability, correcting for tanh transformation
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Calculate the mean action (without exploration noise)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
        
    def evaluate(self, state, action):
        """
        Evaluate an action given a state, useful for PPO to get new log probs for old actions.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: (action, log_prob, entropy) - resampled action, log probability, and entropy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Get log probabilities
        # For actions in tanh-transformed space, we need to apply correction
        action_normalized = (action - self.action_bias) / self.action_scale
        
        # Inverse tanh to get original gaussian samples x_t
        action_normalized = torch.clamp(action_normalized, -0.999, 0.999)  # Avoid numerical instability
        x_t = 0.5 * torch.log((1 + action_normalized) / (1 - action_normalized))
        
        # Get log probability
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh correction: log(1 - tanh^2(x)) = log(1 - y^2)
        log_prob -= torch.log(self.action_scale * (1 - action_normalized.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Compute entropy
        entropy = normal.entropy().sum(1, keepdim=True)
        
        return action, log_prob, entropy

    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            GaussianPolicy: Self reference for chaining
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


# Add these class definitions to your models.py file

class DeterministicPolicy(nn.Module):
    """
    Deterministic policy for SAC-Deterministic or TD3.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        """
        Initialize Deterministic Policy.
        
        Args:
            num_inputs (int): Dimension of the state space
            num_actions (int): Dimension of the action space
            hidden_dim (int): Size of hidden layers
            action_space: Action space with high and low bounds (optional)
        """
        super(DeterministicPolicy, self).__init__()
        
        # Neural network layers
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        
        # Exploration noise
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # Action scaling parameters
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """
        Compute deterministic action.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Mean action
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        """
        Sample actions with exploration noise.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action, log_prob, mean) - action with noise, dummy log prob, and mean
        """
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            DeterministicPolicy: Self reference for chaining
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class DiscretePolicy(nn.Module):
    """
    Policy for discrete action spaces in SAC.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Initialize Discrete Policy.
        
        Args:
            num_inputs (int): Dimension of state space
            num_actions (int): Number of discrete actions
            hidden_dim (int): Size of hidden layers
        """
        super(DiscretePolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output logits for each action
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        
        self.num_actions = num_actions
        self.apply(weights_init_)
        
    def forward(self, state):
        """
        Compute action probabilities for the given state.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Action logits
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action_logits = self.linear3(x)
        
        return action_logits
    
    def sample(self, state):
        """
        Sample actions from the policy.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action, log_prob, action_probs)
        """
        action_logits = self.forward(state)
        
        # Use Gumbel-Softmax for differentiable discrete sampling
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Sample an action
        if len(state.shape) == 1 or state.shape[0] == 1:
            # Single state
            action = action_dist.sample().unsqueeze(-1)
        else:
            # Batch of states
            action = action_dist.sample().unsqueeze(-1)
        
        # Calculate log probability
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_prob = torch.log(action_probs + z)
        log_prob = torch.gather(log_prob, -1, action)
        
        return action, log_prob, action_probs
        
    def evaluate(self, state, action):
        """
        Evaluate an action given a state, useful for PPO to get new log probs for old actions.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor (indices)
            
        Returns:
            tuple: (action, log_prob, entropy) - same action, log probability, and entropy
        """
        action_logits = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Calculate entropy
        entropy = -torch.sum(action_probs * log_probs, dim=-1, keepdim=True)
        
        # Get log probabilities for the given actions
        if action.dim() == 2 and action.size(1) == 1:
            # If actions are indices
            selected_log_probs = torch.gather(log_probs, 1, action.long())
        else:
            # If actions are one-hot encoded
            selected_log_probs = (action * log_probs).sum(dim=-1, keepdim=True)
        
        return action, selected_log_probs, entropy
    
    def to(self, device):
        """
        Move model to specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            DiscretePolicy: Self reference for chaining
        """
        return super(DiscretePolicy, self).to(device)