import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from rl_utils import soft_update, hard_update
from models import QNetwork, GaussianPolicy, DeterministicPolicy, DiscretePolicy

class SAC(object):
    """
    Soft Actor-Critic (SAC) agent implementation.
    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework.
    """
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize the SAC agent.
        
        Args:
            num_inputs (int): Dimension of state space
            action_space: Action space with shape and bounds
            args: Configuration arguments
        """
        # Algorithm parameters
        self.gamma = args.gamma                           # Discount factor
        self.tau = args.tau                               # Soft update coefficient
        self.alpha = args.alpha                           # Temperature parameter for entropy
        self.policy_type = args.policy                    # Policy type (Gaussian or Deterministic)
        self.target_update_interval = args.target_update_interval  # Frequency of target network updates
        self.automatic_entropy_tuning = args.automatic_entropy_tuning  # Whether to tune entropy automatically

        # Device setup
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        
        # Determine action dimensions and type
        if isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
            self.discrete = False
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n  # Number of discrete actions
            self.discrete = True
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        # Critic networks
        self.critic = QNetwork(num_inputs, self.action_dim, args.hidden_size, discrete=self.discrete).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # Target critic network
        self.critic_target = QNetwork(num_inputs, self.action_dim, args.hidden_size, discrete=self.discrete).to(self.device)
        hard_update(self.critic_target, self.critic)  # Initialize target with same weights

        # Policy network based on policy type
        if self.policy_type == "Gaussian":
            # Set up automatic entropy tuning if enabled
            if self.automatic_entropy_tuning:
                # Target entropy is the negative of action dimensions for continuous actions
                # For discrete actions, we use a different formula
                if self.discrete:
                    self.target_entropy = -0.98 * np.log(1.0/self.action_dim)  # Slightly less than max entropy
                else:
                    self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.device)).item()
                    
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            # Create appropriate policy
            if self.discrete:
                self.policy = DiscretePolicy(num_inputs, self.action_dim, args.hidden_size).to(self.device)
            else:
                self.policy = GaussianPolicy(num_inputs, self.action_dim, args.hidden_size, action_space).to(self.device)
                
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        else:  # "Deterministic" policy
            self.alpha = 0
            self.automatic_entropy_tuning = False
            if self.discrete:
                raise ValueError("Deterministic policy is not supported for discrete actions")
            self.policy = DeterministicPolicy(num_inputs, self.action_dim, args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state
            evaluate (bool): Whether to evaluate (use mean) or explore (sample)
            
        Returns:
            numpy.ndarray: Selected action
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if self.discrete:
            # Discrete actions
            with torch.no_grad():
                if evaluate:
                    # Use most probable action for evaluation
                    _, _, action_probs = self.policy.sample(state)
                    action = torch.argmax(action_probs, dim=-1, keepdim=True)
                else:
                    # Sample action for exploration
                    action, _, _ = self.policy.sample(state)
            
            return action.detach().cpu().numpy()[0]
        else:
            # Continuous actions
            if evaluate is False:
                # Sample from the policy for exploration
                action, _, _ = self.policy.sample(state)
            else:
                # Use the mean action for evaluation
                _, _, action = self.policy.sample(state)
                
            return action.detach().cpu().numpy()[0]

    def select_actions_vec(self, states, evaluate=False):
        """
        Select actions for multiple states (vectorized version).
        
        Args:
            states: Batch of states
            evaluate (bool): Whether to evaluate (use mean) or explore (sample)
            
        Returns:
            numpy.ndarray: Batch of selected actions
        """
        # Handle empty states case (can happen during evaluation)
        if states.size == 0 or len(states.shape) < 2:
            return np.array([])
        
        # Convert to tensor
        states = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            if self.discrete:
                # Discrete actions
                if evaluate:
                    # Use most probable action for evaluation
                    _, _, action_probs = self.policy.sample(states)
                    actions = torch.argmax(action_probs, dim=-1, keepdim=True)
                else:
                    # Sample action for exploration
                    actions, _, _ = self.policy.sample(states)
            else:
                # Continuous actions
                if evaluate is False:
                    # Sample from the policy for exploration
                    actions, _, _ = self.policy.sample(states)
                else:
                    # Use the mean action for evaluation
                    _, _, actions = self.policy.sample(states)
                
        return actions.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        """
        Update policy and value parameters using batch from replay buffer.
        
        Args:
            memory: Replay buffer
            batch_size (int): Size of the batch
            updates (int): Number of updates performed so far
            
        Returns:
            tuple: Loss values for logging
        """
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        
        if self.discrete:
            # Discrete actions need to be long tensors for indexing
            action_batch = torch.LongTensor(action_batch).to(self.device)
        else:
            # Continuous actions
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Update critic networks
        with torch.no_grad():
            if self.discrete:
                # For discrete actions
                # Sample next action and compute Q-values
                next_state_action, next_state_log_pi, next_state_probs = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
                
                # Take expectation over actions
                next_state_log_pi = torch.sum(next_state_probs * next_state_log_pi, dim=1, keepdim=True)
                
                # Compute the soft value
                next_q1_target = torch.sum(next_state_probs * qf1_next_target, dim=1, keepdim=True)
                next_q2_target = torch.sum(next_state_probs * qf2_next_target, dim=1, keepdim=True)
                min_qf_next_target = torch.min(next_q1_target, next_q2_target) - self.alpha * next_state_log_pi
            else:
                # For continuous actions
                # Sample next action and compute Q-values
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                
                # Take minimum of Q-values to mitigate positive bias
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            
            # Compute target Q-value
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
            
        # Current Q-values
        if self.discrete:
            # Get all Q-values and then select for actions taken
            qf1, qf2 = self.critic(state_batch)
            qf1 = torch.gather(qf1, 1, action_batch)
            qf2 = torch.gather(qf2, 1, action_batch)
        else:
            # Continuous actions
            qf1, qf2 = self.critic(state_batch, action_batch)
        
        # Compute critic loss
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Update critics
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update policy
        if self.discrete:
            # For discrete actions
            # Get action probabilities
            action_logits = self.policy(state_batch)
            action_probs = F.softmax(action_logits, dim=-1)
            log_action_probs = F.log_softmax(action_logits, dim=-1)
            
            # Get Q-values for all actions
            with torch.no_grad():
                qf1, qf2 = self.critic(state_batch)
                min_qf = torch.min(qf1, qf2)
            
            # KL divergence term
            if self.automatic_entropy_tuning:
                entropy = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
                entropy_term = self.alpha * entropy
            else:
                entropy_term = 0
            
            # Compute the expected Q-value
            expected_q = torch.sum(action_probs * min_qf, dim=1, keepdim=True)
            
            # Policy loss with entropy regularization
            policy_loss = torch.mean(entropy_term - expected_q)
        else:
            # For continuous actions
            pi, log_pi, _ = self.policy.sample(state_batch)
            
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            
            # Policy loss with entropy regularization
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update temperature parameter alpha if automatic tuning is enabled
        if self.automatic_entropy_tuning:
            if self.discrete:
                # For discrete actions
                # Current entropy
                entropy = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True).mean()
                
                # Alpha loss
                alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            else:
                # For continuous actions
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For logging
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For logging

        # Soft update target networks
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        """
        Save agent parameters to a checkpoint file.
        
        Args:
            env_name (str): Environment name for checkpoint filename
            suffix (str): Additional suffix for checkpoint filename
            ckpt_path (str): Path to save checkpoint, if None uses default
        """
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
            
        if ckpt_path is None:
            ckpt_path = f"checkpoints/sac_checkpoint_{env_name}_{suffix}"
            
        print(f'Saving models to {ckpt_path}')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict()
        }, ckpt_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        """
        Load agent parameters from a checkpoint file.
        
        Args:
            ckpt_path (str): Path to checkpoint file
            evaluate (bool): Whether to set networks to evaluation mode
        """
        print(f'Loading models from {ckpt_path}')
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()