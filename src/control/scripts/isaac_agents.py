import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
<<<<<<<< HEAD:src/control/scripts/isaac_agents.py
from isaac_utils import soft_update, hard_update
from isaac_models import QNetwork, GaussianPolicy, DeterministicPolicy, ActorCritic
========
import gymnasium as gym
from rl_utils import soft_update, hard_update
from models import GaussianPolicy, DeterministicPolicy, DiscretePolicy, ValueNetwork, QNetwork
>>>>>>>> alternate_timeline:src/control/scripts/agents.py

class SAC(object):
    """
    Soft Actor-Critic (SAC) agent implementation.
    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework.
    
    Can be configured to use single or twin critic, and different entropy approaches.
    """
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize the SAC agent.
        
        Args:
            num_inputs (int): Dimension of state space
            action_space: Action space with shape and bounds
            args: Configuration arguments
                Can include additional fields:
                - use_twin_critic: Whether to use twin critic (True) or single critic (False)
                - entropy_mode: "none", "fixed", or "adaptive"
        """
        # Algorithm parameters
        self.gamma = args.gamma                           # Discount factor
        self.tau = args.tau                               # Soft update coefficient
        self.alpha = args.alpha                           # Temperature parameter for entropy
        self.policy_type = args.policy                    # Policy type (Gaussian or Deterministic)
        self.target_update_interval = args.target_update_interval  # Frequency of target network updates
        
        # Special configuration parameters with defaults for backward compatibility
        self.use_twin_critic = getattr(args, 'use_twin_critic', True)  # Default to twin critic
        self.entropy_mode = getattr(args, 'entropy_mode', 'adaptive')  # Default to adaptive entropy
        self.automatic_entropy_tuning = (self.entropy_mode == 'adaptive')
        
        # If entropy mode is "none", set alpha to 0
        if self.entropy_mode == "none":
            self.alpha = 0.0
            self.automatic_entropy_tuning = False
        elif self.entropy_mode == "fixed":
            # Use provided alpha value
            self.automatic_entropy_tuning = False

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
                
                if self.use_twin_critic:
                    min_qf_next_target = torch.min(next_q1_target, next_q2_target) - self.alpha * next_state_log_pi
                else:
                    # Use only the first critic for single critic mode
                    min_qf_next_target = next_q1_target - self.alpha * next_state_log_pi
            else:
                # For continuous actions
                # Sample next action and compute Q-values
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                
                # Take minimum of Q-values if using twin critic, otherwise use first critic
                if self.use_twin_critic:
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                else:
                    min_qf_next_target = qf1_next_target - self.alpha * next_state_log_pi
            
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
        
        # Compute critic loss (use both critics or just the first one)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        if self.use_twin_critic:
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss
        else:
            qf2_loss = torch.tensor(0.)  # Dummy value for single critic
            qf_loss = qf1_loss

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
                if self.use_twin_critic:
                    min_qf = torch.min(qf1, qf2)
                else:
                    min_qf = qf1
            
            # KL divergence term (entropy)
            entropy = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
            entropy_term = self.alpha * entropy  # Will be 0 if entropy_mode is "none"
            
            # Compute the expected Q-value
            expected_q = torch.sum(action_probs * min_qf, dim=1, keepdim=True)
            
            # Policy loss with entropy regularization
            policy_loss = torch.mean(entropy_term - expected_q)
        else:
            # For continuous actions
            pi, log_pi, _ = self.policy.sample(state_batch)
            
            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            if self.use_twin_critic:
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
            else:
                min_qf_pi = qf1_pi
            
            # Policy loss with entropy regularization
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update temperature parameter alpha if adaptive entropy tuning is enabled
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
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'use_twin_critic': self.use_twin_critic,
            'entropy_mode': self.entropy_mode,
            'alpha': self.alpha,
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
            
            # Load configuration if available
            self.use_twin_critic = checkpoint.get('use_twin_critic', True)
            self.entropy_mode = checkpoint.get('entropy_mode', 'adaptive')
            self.alpha = checkpoint.get('alpha', self.alpha)

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

class PPOCLIP(object):
    """
    Proximal Policy Optimization (PPO) with clipped objective.
    PPO is an on-policy actor-critic algorithm that uses a clipped surrogate objective
    to limit policy updates, preventing too large policy changes.
    """
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize PPO with clipped objective.
        
        Args:
            num_inputs (int): Dimension of state space
            action_space: Action space with shape and bounds
            args: Configuration arguments
        """
        # Algorithm parameters
        self.gamma = args.gamma                   # Discount factor
        self.tau = args.tau                       # Soft update coefficient (if needed)
        self.clip_param = args.clip_param         # Clipping parameter
        self.ppo_epoch = args.ppo_epoch           # Number of optimization epochs
        self.num_mini_batch = args.num_mini_batch # Number of minibatches for optimization
        self.value_loss_coef = args.value_loss_coef  # Value loss coefficient
        self.entropy_coef = args.entropy_coef     # Entropy coefficient
        self.max_grad_norm = args.max_grad_norm   # Maximum gradient norm
        self.use_clipped_value_loss = args.use_clipped_value_loss  # Whether to use clipped value loss

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

        # Value network (critic)
        self.value_net = ValueNetwork(num_inputs, args.hidden_size).to(device=self.device)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=args.value_lr)

        # Policy network (actor)
        if self.discrete:
            self.policy = DiscretePolicy(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        else:
            self.policy = GaussianPolicy(num_inputs, self.action_dim, args.hidden_size, action_space).to(self.device)
            
        self.policy_optimizer = Adam(self.policy.parameters(), lr=args.policy_lr)

    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state
            evaluate (bool): Whether to evaluate (use mean) or explore (sample)
            
        Returns:
            numpy.ndarray: Selected action
            float: Log probability of the action
            float: Value estimate
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # Get value estimate
        value = self.value_net(state)
        
        if self.discrete:
            # Discrete actions
            with torch.no_grad():
                if evaluate:
                    # Use most probable action for evaluation
                    _, _, action_probs = self.policy.sample(state)
                    action = torch.argmax(action_probs, dim=-1, keepdim=True)
                    log_prob = torch.log(action_probs.gather(-1, action))
                else:
                    # Sample action for training
                    action, log_prob, _ = self.policy.sample(state)
            
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]
        else:
            # Continuous actions
            with torch.no_grad():
                if evaluate:
                    # Use mean action for evaluation
                    _, _, action = self.policy.sample(state)
                    # For evaluation, we still need the log probability
                    _, log_prob, _ = self.policy.evaluate(state, action)
                else:
                    # Sample action for training
                    action, log_prob, _ = self.policy.sample(state)
                
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def select_actions_vec(self, states, evaluate=False):
        """
        Select actions for multiple states (vectorized version).
        
        Args:
            states: Batch of states
            evaluate (bool): Whether to evaluate (use mean) or explore (sample)
            
        Returns:
            numpy.ndarray: Batch of selected actions
            numpy.ndarray: Batch of log probabilities
            numpy.ndarray: Batch of value estimates
        """
        # Handle empty states case (can happen during evaluation)
        if states.size == 0 or len(states.shape) < 2:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to tensor
        states = torch.FloatTensor(states).to(self.device)
        
        # Get value estimates
        values = self.value_net(states)
        
        with torch.no_grad():
            if self.discrete:
                # Discrete actions
                if evaluate:
                    # Use most probable action for evaluation
                    _, _, action_probs = self.policy.sample(states)
                    actions = torch.argmax(action_probs, dim=-1, keepdim=True)
                    log_probs = torch.log(action_probs.gather(-1, actions))
                else:
                    # Sample action for training
                    actions, log_probs, _ = self.policy.sample(states)
            else:
                # Continuous actions
                if evaluate:
                    # Use mean action for evaluation
                    _, _, actions = self.policy.sample(states)
                    # For evaluation, we still need the log probabilities
                    _, log_probs, _ = self.policy.evaluate(states, actions)
                else:
                    # Sample action for training
                    actions, log_probs, _ = self.policy.sample(states)
                
        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), values.detach().cpu().numpy()

    def get_value(self, state):
        """
        Get value estimate for a state.
        
        Args:
            state: Current state
            
        Returns:
            float: Value estimate
        """
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            value = self.value_net(state)
        return value.detach().cpu().numpy()

    def update_parameters(self, rollouts):
        """
        Update policy and value parameters using PPO with clipped objective.
        
        Args:
            rollouts: RolloutStorage containing batch of experiences
            
        Returns:
            tuple: Loss values for logging
        """
        # Get rollout data
        rollout_data = rollouts.get_data()
        states = torch.FloatTensor(rollout_data['states']).to(self.device)
        actions = torch.FloatTensor(rollout_data['actions']).to(self.device)
        returns = torch.FloatTensor(rollout_data['returns']).to(self.device).view(-1, 1)
        masks = torch.FloatTensor(rollout_data['masks']).to(self.device).view(-1, 1)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device).view(-1, 1)
        advantages = torch.FloatTensor(rollout_data['advantages']).to(self.device).view(-1, 1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert discrete actions to long tensors for indexing if needed
        if self.discrete and actions.shape[-1] == 1:
            actions = actions.long()
        
        # Training info for logging
        value_loss_epoch = 0
        policy_loss_epoch = 0
        entropy_epoch = 0
        clip_fraction_epoch = 0
        
        # Create batches for multiple epochs of training on the same data
        batch_size = states.size(0)
        mini_batch_size = batch_size // self.num_mini_batch
        
        for _ in range(self.ppo_epoch):
            # Generate random indices for creating mini-batches
            indices = torch.randperm(batch_size).to(self.device)
            
            for start_idx in range(0, batch_size, mini_batch_size):
                # Extract mini-batch indices
                end_idx = min(start_idx + mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_masks = masks[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Evaluate actions and get current log probs and entropy
                if self.discrete:
                    # For discrete actions
                    action_logits = self.policy(mb_states)
                    action_probs = F.softmax(action_logits, dim=-1)
                    log_probs = F.log_softmax(action_logits, dim=-1)
                    
                    if mb_actions.shape[-1] == 1:  # If actions are indices
                        # Calculate entropy - ensure it's a scalar by taking mean
                        dist_entropy = (-torch.sum(action_probs * log_probs, dim=-1)).mean()
                        mb_new_log_probs = torch.gather(log_probs, 1, mb_actions)
                    else:  # If actions are one-hot encoded
                        # Calculate entropy - ensure it's a scalar by taking mean
                        dist_entropy = (-torch.sum(action_probs * log_probs, dim=-1)).mean()
                        mb_new_log_probs = (mb_actions * log_probs).sum(dim=-1, keepdim=True)
                else:
                    # For continuous actions
                    _, mb_new_log_probs, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                    # Ensure entropy is a scalar
                    if dist_entropy.dim() > 0:
                        dist_entropy = dist_entropy.mean()
                
                # Get current value prediction
                values = self.value_net(mb_states)
                
                # Calculate ratios and surrogate objectives
                ratios = torch.exp(mb_new_log_probs - mb_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Record clipping statistics
                clipped = (ratios < 1.0 - self.clip_param) | (ratios > 1.0 + self.clip_param)
                clip_fraction = clipped.float().mean().item()
                clip_fraction_epoch += clip_fraction
                
                # Value loss
                if self.use_clipped_value_loss:
                    # Get old value predictions (assuming they are stored in rollouts)
                    old_values = rollout_data['values']
                    old_values = torch.FloatTensor(old_values).to(self.device).view(-1, 1)[mb_indices]
                    
                    # Clipped value loss
                    values_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
                    value_loss_unclipped = (values - mb_returns).pow(2)
                    value_loss_clipped = (values_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    # Simple MSE value loss
                    value_loss = 0.5 * F.mse_loss(values, mb_returns)
                
                # Combined loss with value and entropy terms
                # Make sure each component is a scalar
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * dist_entropy
                
                # Make sure loss is a scalar before calling backward()
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Update policy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Accumulate epoch statistics - ensure we're getting scalar values
                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                # Convert entropy to scalar before adding to epoch total
                entropy_epoch += dist_entropy.item()
        
        # Calculate average statistics
        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        policy_loss_epoch /= num_updates
        entropy_epoch /= num_updates
        clip_fraction_epoch /= num_updates
        
        return value_loss_epoch, policy_loss_epoch, entropy_epoch, clip_fraction_epoch
    
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
            ckpt_path = f"checkpoints/ppo_clip_checkpoint_{env_name}_{suffix}"
            
        print(f'Saving models to {ckpt_path}')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
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
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.value_net.eval()
            else:
                self.policy.train()
                self.value_net.train()

class PPOKL(object):
    """
    Proximal Policy Optimization (PPO) with KL-divergence constraint.
    PPO-KL uses an adaptive KL-divergence penalty to prevent too large policy changes.
    """
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize PPO with KL-divergence constraint.
        
        Args:
            num_inputs (int): Dimension of state space
            action_space: Action space with shape and bounds
            args: Configuration arguments
        """
        # Algorithm parameters
        self.gamma = args.gamma                   # Discount factor
        self.tau = args.tau                       # Soft update coefficient (if needed)
        self.ppo_epoch = args.ppo_epoch           # Number of optimization epochs
        self.num_mini_batch = args.num_mini_batch # Number of minibatches for optimization
        self.value_loss_coef = args.value_loss_coef  # Value loss coefficient
        self.entropy_coef = args.entropy_coef     # Entropy coefficient
        self.max_grad_norm = args.max_grad_norm   # Maximum gradient norm
        self.use_clipped_value_loss = args.use_clipped_value_loss # Whether to use clipped value loss
        self.clip_param = args.clip_param         # Clipping parameter (from PPO-CLIP)


        # KL-specific parameters
        self.kl_target = args.kl_target           # Target KL-divergence
        self.kl_beta = args.kl_beta               # Initial KL coefficient
        self.kl_adaptive = args.kl_adaptive       # Whether to adapt KL coefficient
        self.kl_cutoff_factor = args.kl_cutoff_factor  # KL cutoff factor
        self.kl_cutoff_coef = args.kl_cutoff_coef      # KL cutoff coefficient
        self.min_kl_coef = args.min_kl_coef       # Minimum KL coefficient
        self.max_kl_coef = args.max_kl_coef       # Maximum KL coefficient


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

        # Value network (critic)
        self.value_net = ValueNetwork(num_inputs, args.hidden_size).to(device=self.device)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=args.value_lr)

        # Policy network (actor)
        if self.discrete:
            self.policy = DiscretePolicy(num_inputs, self.action_dim, args.hidden_size).to(self.device)
        else:
            self.policy = GaussianPolicy(num_inputs, self.action_dim, args.hidden_size, action_space).to(self.device)
            
        self.policy_optimizer = Adam(self.policy.parameters(), lr=args.policy_lr)

    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state
            evaluate (bool): Whether to evaluate (use mean) or explore (sample)
            
        Returns:
            numpy.ndarray: Selected action
            float: Log probability of the action
            float: Value estimate
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # Get value estimate
        value = self.value_net(state)
        
        if self.discrete:
            # Discrete actions
            with torch.no_grad():
                if evaluate:
                    # Use most probable action for evaluation
                    _, _, action_probs = self.policy.sample(state)
                    action = torch.argmax(action_probs, dim=-1, keepdim=True)
                    log_prob = torch.log(action_probs.gather(-1, action))
                else:
                    # Sample action for training
                    action, log_prob, _ = self.policy.sample(state)
            
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]
        else:
            # Continuous actions
            with torch.no_grad():
                if evaluate:
                    # Use mean action for evaluation
                    _, _, action = self.policy.sample(state)
                    # For evaluation, we still need the log probability
                    _, log_prob, _ = self.policy.evaluate(state, action)
                else:
                    # Sample action for training
                    action, log_prob, _ = self.policy.sample(state)
                
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0]

    def select_actions_vec(self, states, evaluate=False):
        """
        Select actions for multiple states (vectorized version).
        
        Args:
            states: Batch of states
            evaluate (bool): Whether to evaluate (use mean) or explore (sample)
            
        Returns:
            numpy.ndarray: Batch of selected actions
            numpy.ndarray: Batch of log probabilities
            numpy.ndarray: Batch of value estimates
        """
        # Handle empty states case (can happen during evaluation - gave me an error at 99%)
        if states.size == 0 or len(states.shape) < 2:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to tensor
        states = torch.FloatTensor(states).to(self.device)
        
        # Get value estimates
        values = self.value_net(states)
        
        with torch.no_grad():
            if self.discrete:
                # Discrete actions
                if evaluate:
                    # Use most probable action for evaluation
                    _, _, action_probs = self.policy.sample(states)
                    actions = torch.argmax(action_probs, dim=-1, keepdim=True)
                    log_probs = torch.log(action_probs.gather(-1, actions))
                else:
                    # Sample action for training
                    actions, log_probs, _ = self.policy.sample(states)
            else:
                # Continuous actions
                if evaluate:
                    # Use mean action for evaluation
                    _, _, actions = self.policy.sample(states)
                    # For evaluation, we still need the log probabilities
                    _, log_probs, _ = self.policy.evaluate(states, actions)
                else:
                    # Sample action for training
                    actions, log_probs, _ = self.policy.sample(states)
                
        return actions.detach().cpu().numpy(), log_probs.detach().cpu().numpy(), values.detach().cpu().numpy()

    def get_value(self, state):
        """
        Get value estimate for a state.
        
        Args:
            state: Current state
            
        Returns:
            float: Value estimate
        """
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            value = self.value_net(state)
        return value.detach().cpu().numpy()

    def update_parameters(self, rollouts):
        """
        Update policy and value parameters using collected rollouts with KL penalty.
        """
        # Get rollout data
        rollout_data = rollouts.get_data()
        
        if len(rollout_data.get('states', [])) == 0:
            return 0.0, 0.0, 0.0, 0.0, self.kl_beta
        
        # Convert data to tensors
        states = torch.FloatTensor(rollout_data['states']).to(self.device)
        actions = torch.FloatTensor(rollout_data['actions']).to(self.device)
        returns = torch.FloatTensor(rollout_data['returns']).to(self.device).view(-1, 1)
        old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device).view(-1, 1)
        advantages = torch.FloatTensor(rollout_data['advantages']).to(self.device).view(-1, 1)
        
        # Normalize advantages
        if advantages.numel() > 1:
            adv_mean, adv_std = advantages.mean(), advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - adv_mean) / adv_std
        
        if self.discrete and actions.shape[-1] == 1:
            actions = actions.long()
        
        # Training tracking
        value_loss_epoch = 0
        policy_loss_epoch = 0
        entropy_epoch = 0
        kl_divergence_epoch = 0
        
        batch_size = states.size(0)
        mini_batch_size = max(1, batch_size // self.num_mini_batch)
        
        # Get initial policy distribution for all states once
        with torch.no_grad():
            if self.discrete:
                all_old_logits = self.policy(states)
                all_old_probs = F.softmax(all_old_logits, dim=-1).detach()
            else:
                all_old_means, all_old_log_stds = self.policy(states)
                all_old_means = all_old_means.detach()
                all_old_stds = all_old_log_stds.exp().detach()
        
        # Training loop
        for epoch in range(self.ppo_epoch):
            indices = torch.randperm(batch_size).to(self.device)
            epoch_kl = 0
            num_batches = 0
            
            for start_idx in range(0, batch_size, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Get batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Forward pass
                values = self.value_net(mb_states)
                
                if self.discrete:
                    # Discrete actions
                    mb_old_probs = all_old_probs[mb_indices].detach()
                    logits = self.policy(mb_states)
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get action log probabilities
                    if mb_actions.dim() > 1 and mb_actions.size(1) == 1:
                        mb_new_log_probs = torch.gather(log_probs, 1, mb_actions)
                    else:
                        mb_new_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1))
                    
                    entropy = -(probs * log_probs).sum(-1).mean()
                    
                    # KL divergence
                    kl = mb_old_probs * (torch.log(mb_old_probs + 1e-10) - torch.log(probs + 1e-10))
                    kl_divergence = kl.sum(-1).mean()
                    
                    if kl_divergence < 1e-6:
                        kl_divergence = torch.tensor(1e-6, device=self.device)
                else:
                    # Continuous actions
                    mb_old_means = all_old_means[mb_indices].detach()
                    mb_old_stds = all_old_stds[mb_indices].detach()
                    
                    mean, log_std = self.policy(mb_states)
                    std = log_std.exp()
                    
                    normal = torch.distributions.Normal(mean, std)
                    mb_new_log_probs = normal.log_prob(mb_actions).sum(-1, keepdim=True)
                    
                    entropy = normal.entropy().sum(-1).mean()
                    
                    # Improved KL calculation
                    var_ratio = (mb_old_stds / (std + 1e-10)).pow(2)
                    mean_diff_term = ((mb_old_means - mean) / (std + 1e-10)).pow(2)
                    log_std_diff = 2 * (log_std - torch.log(mb_old_stds + 1e-10))
                    
                    kl_div = 0.5 * (var_ratio + mean_diff_term - 1.0 - log_std_diff)
                    kl_divergence = kl_div.sum(-1).mean()
                    
                    # Fallback for numerical stability
                    if kl_divergence < 1e-6:
                        std_diff = (std - mb_old_stds).pow(2).mean()
                        mean_diff = (mean - mb_old_means).pow(2).mean()
                        kl_divergence = 0.5 * (mean_diff + std_diff) + 1e-6
                
                # Calculate ratio and loss
                ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                ratio = torch.clamp(ratio, 0.1, 10.0)
                surrogate = ratio * mb_advantages
                kl_penalty = self.kl_beta * kl_divergence
                
                # Policy loss with KL penalty and entropy bonus
                policy_loss = -surrogate.mean() + kl_penalty - self.entropy_coef * entropy
                
                # Value loss
                if self.use_clipped_value_loss:
                    old_values = self.value_net(mb_states).detach()
                    values_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
                    value_loss_unclipped = (values - mb_returns).pow(2)
                    value_loss_clipped = (values_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(values, mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss
                
                if torch.isfinite(loss).all():
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    loss.backward()
                    
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                    
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                
                # Track statistics
                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                entropy_epoch += entropy.item()
                kl_divergence_epoch += kl_divergence.item()
                
                epoch_kl += kl_divergence.item()
                num_batches += 1
            
            # Adapt KL coefficient (with wider adjustment range)
            if self.kl_adaptive and num_batches > 0:
                avg_kl = epoch_kl / num_batches
                
                # More aggressive adjustment
                if avg_kl < self.kl_target / 1.2:  # Changed from 1.5 to 1.2
                    self.kl_beta = max(self.min_kl_coef / 10.0, self.kl_beta / 2.0)  # More aggressive decrease
                elif avg_kl > self.kl_target * 1.2:  # Changed from 1.5 to 1.2
                    self.kl_beta = min(self.max_kl_coef, self.kl_beta * 2.0)  # More aggressive increase
        
        # Calculate average statistics
        num_updates = max(1, self.ppo_epoch * num_batches)
        value_loss_epoch /= num_updates
        policy_loss_epoch /= num_updates
        entropy_epoch /= num_updates
        kl_divergence_epoch /= num_updates
        
        return value_loss_epoch, policy_loss_epoch, entropy_epoch, kl_divergence_epoch, self.kl_beta
        
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
            ckpt_path = f"checkpoints/ppo_kl_checkpoint_{env_name}_{suffix}"
            
        print(f'Saving models to {ckpt_path}')
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'kl_beta': self.kl_beta
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
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            
            # Load KL coefficient if present
            if 'kl_beta' in checkpoint:
                self.kl_beta = checkpoint['kl_beta']

            if evaluate:
                self.policy.eval()
                self.value_net.eval()
            else:
                self.policy.train()
                self.value_net.train()