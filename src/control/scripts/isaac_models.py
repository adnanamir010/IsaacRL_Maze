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

class QNetwork(nn.Module):
    """
    Critic network for SAC that estimates Q-values (state-action values).
    Implements the double Q-network architecture.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Initialize Q-Network.
        
        Args:
            num_inputs (int): Dimension of the state space
            num_actions (int): Dimension of the action space
            hidden_dim (int): Size of hidden layers
        """
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear3 = nn.Linear(int(hidden_dim/2), 1)

        # Q2 architecture (for double Q-learning)
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear6 = nn.Linear(int(hidden_dim/2), 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        """
        Compute Q-values for given state-action pairs.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: (Q1 value, Q2 value)
        """
        # Concatenate state and action as input
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
    
    def Q1(self, state, action):
        """
        Returns only the first Q-value, useful for computing actor loss.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            torch.Tensor: Q1 value
        """
        # Concatenate state and action
        xu = torch.cat([state, action], 1)
        
        # Forward pass through first Q-network only
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        return x1


class GaussianPolicy(nn.Module):
    """
    Stochastic policy that outputs a Gaussian distribution over actions.
    Uses GRU layers for sequence modeling capabilities.
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
        
        # GRU layers for sequence modeling
        self.gru1 = nn.GRU(num_inputs, hidden_dim)
        self.gru2 = nn.GRU(hidden_dim, int(hidden_dim/2))

        # Policy head for mean and log standard deviation
        self.mean_linear = nn.Linear(int(hidden_dim/2), num_actions)
        self.log_std_linear = nn.Linear(int(hidden_dim/2), num_actions)

        self.apply(weights_init_)

        # Action scaling parameters
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            high = np.ones(action_space.shape[0])
            low = -1 * np.ones(action_space.shape[0])

            self.action_scale = torch.FloatTensor((high - low) / 2.)
            self.action_bias = torch.FloatTensor((high + low) / 2.)

    def forward(self, state):
        """
        Compute action distribution parameters.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (mean, log_std) of action distribution
        """
        # Process through GRU layers
        x, _ = self.gru1(state)
        x = F.relu(x)
        x, _ = self.gru2(x)
        x = F.relu(x)
        
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


class ValueNetwork(nn.Module):
    """
    Value network for PPO or other algorithms that need state value estimates.
    """
    def __init__(self, num_inputs, hidden_dim):
        """
        Initialize Value Network.
        
        Args:
            num_inputs (int): Dimension of the state space
            hidden_dim (int): Size of hidden layers
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        """
        Compute state value.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: State value
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        """
        Initialize Actor-Critic Network.
        
        Args:
            num_inputs (int): Dimension of the state space
            num_actions (int): Dimension of the action space
            hidden_dim (int): Size of hidden layers
            action_space: Action space with high and low bounds (optional)
        """
        super(ActorCritic, self).__init__()
        
        # Actor (policy) network
        self.actor = GaussianPolicy(num_inputs, num_actions, hidden_dim, action_space)
        
        # Critic (value) network
        self.critic = ValueNetwork(num_inputs, hidden_dim)
        
    def forward(self, state):
        """
        Compute both policy and value.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action_mean, action_log_std, state_value)
        """
        mean, log_std = self.actor(state)
        value = self.critic(state)
        
        return mean, log_std, value
        
    def act(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state (torch.Tensor): State tensor
            deterministic (bool): Whether to use deterministic action (mean)
            
        Returns:
            tuple: (action, log_prob, entropy, state_value)
        """
        mean, log_std = self.actor(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            # Sample action
            action = normal.sample()
            
        # Compute log probability and entropy
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        entropy = normal.entropy().sum(-1, keepdim=True)
        
        # Apply tanh and scale
        action = torch.tanh(action) * self.actor.action_scale + self.actor.action_bias
        
        # Compute value
        value = self.critic(state)
        
        return action, log_prob, entropy, value