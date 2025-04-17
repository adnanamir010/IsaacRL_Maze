import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from isaac_utils import soft_update, hard_update
from isaac_models import QNetwork, GaussianPolicy, DeterministicPolicy, ActorCritic

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Critic networks
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        # Target critic network
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)  # Initialize target with same weights

        # Policy network based on policy type
        if self.policy_type == "Gaussian":
            # Set up automatic entropy tuning if enabled
            if self.automatic_entropy_tuning:
                # Target entropy is the negative of action dimensions
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            # Create Gaussian policy
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        else:  # "Deterministic" policy
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
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
        
        if evaluate is False:
            # Sample from the policy for exploration
            action, _, _ = self.policy.sample(state)
        else:
            # Use the mean action for evaluation
            _, _, action = self.policy.sample(state)
            
        return action.detach().cpu().numpy()[0]

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
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Update critic networks
        with torch.no_grad():
            # Sample next action and compute Q-values
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            # Take minimum of Q-values to mitigate positive bias
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            
            # Compute target Q-value
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
            
        # Current Q-values
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
            checkpoint = torch.load(ckpt_path)
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


class PPOCLIP(object):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    PPO is an on-policy algorithm that uses a clipped surrogate objective to
    ensure stable policy updates.
    """
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize the PPO agent.
        
        Args:
            num_inputs (int): Dimension of state space
            action_space: Action space with shape and bounds
            args: Configuration arguments
        """
        # Algorithm parameters
        self.gamma = args.gamma                   # Discount factor
        self.gae_lambda = args.gae_lambda         # GAE parameter
        self.clip_param = args.clip_param         # PPO clip parameter
        self.ppo_epochs = args.ppo_epochs         # Number of PPO epochs
        self.num_mini_batch = args.num_mini_batch # Number of minibatches for PPO
        self.value_loss_coef = args.value_loss_coef     # Value loss coefficient
        self.entropy_coef = args.entropy_coef           # Entropy coefficient
        self.max_grad_norm = args.max_grad_norm         # Max gradient norm for clipping
        self.use_clipped_value_loss = args.use_clipped_value_loss  # Whether to use clipped value loss
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic network
        self.actor_critic = ActorCritic(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.optimizer = Adam(self.actor_critic.parameters(), lr=args.lr)
        
    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state
            evaluate (bool): Whether to evaluate (deterministic) or explore (stochastic)
            
        Returns:
            tuple: (action, value, log_prob, entropy)
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # Get action, log probability, entropy and value
        action, log_prob, entropy, value = self.actor_critic.act(state, deterministic=evaluate)
        
        # Convert all tensors to numpy for storage in RolloutStorage
        action_np = action.detach().cpu().numpy()[0]
        value_np = value.detach().cpu().numpy()[0]
        log_prob_np = log_prob.detach().cpu().numpy()[0]
        entropy_np = entropy.detach().cpu().numpy()[0]
        
        return action_np, value_np, log_prob_np, entropy_np

    def update_parameters(self, rollouts):
        """
        Update policy and value parameters using collected rollouts.
        
        Args:
            rollouts: Collected trajectories (either a RolloutStorage object or a dictionary)
            
        Returns:
            dict: Loss values and metrics for logging
        """
        # Get data dict if rollouts is a RolloutStorage object
        if hasattr(rollouts, 'get_data'):
            rollouts = rollouts.get_data()
            
        # Calculate advantages if not already present in rollouts
        if 'advantages' not in rollouts:
            advantages = self._compute_advantages(rollouts)
        else:
            advantages = rollouts['advantages']
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_epoch = 0
        
        # Perform multiple epochs of updates
        for _ in range(self.ppo_epochs):
            # Generate random permutation for mini-batches
            indices = np.random.permutation(len(rollouts['states']))
            batch_size = len(indices) // self.num_mini_batch
            
            # Process each mini-batch
            for start_idx in range(0, len(indices), batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                states_batch = torch.FloatTensor(rollouts['states'][batch_indices]).to(self.device)
                actions_batch = torch.FloatTensor(rollouts['actions'][batch_indices]).to(self.device)
                old_log_probs_batch = torch.FloatTensor(rollouts['log_probs'][batch_indices]).to(self.device)
                returns_batch = torch.FloatTensor(rollouts['returns'][batch_indices]).to(self.device)
                advantages_batch = advantages[batch_indices]
                
                # Forward pass
                mean, log_std, values = self.actor_critic(states_batch)
                dist = torch.distributions.Normal(mean, log_std.exp())
                
                # Get new action log probabilities
                new_log_probs = dist.log_prob(actions_batch).sum(-1, keepdim=True)
                entropy = dist.entropy().mean()
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
                action_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.use_clipped_value_loss:
                    # Clipped value loss
                    value_pred_clipped = rollouts['values'][batch_indices] + \
                                         (values - rollouts['values'][batch_indices]).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    # Standard value loss
                    value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                
                # Overall loss
                loss = value_loss * self.value_loss_coef + action_loss - entropy * self.entropy_coef
                
                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                # Record metrics
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_epoch += entropy.item()
        
        # Average metrics over epochs and mini-batches
        num_updates = self.ppo_epochs * (len(indices) // batch_size)
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_epoch /= num_updates
        
        return {
            'value_loss': value_loss_epoch,
            'action_loss': action_loss_epoch,
            'entropy': entropy_epoch
        }
    
    def _compute_advantages(self, rollouts, next_value=None):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rollouts: Collected trajectories
            next_value: Value estimate for the state after the last stored state.
                        If None, uses the last value in rollouts.
            
        Returns:
            torch.Tensor: Computed advantages
        """
        # Convert to tensors
        rewards = torch.FloatTensor(rollouts['rewards']).to(self.device)
        values = torch.FloatTensor(rollouts['values']).to(self.device)
        masks = torch.FloatTensor(rollouts['masks']).to(self.device)
        
        # For the last value, use provided next_value or the last value in rollouts
        if next_value is None:
            next_values = torch.cat([values[1:], values[-1:]], 0)
        else:
            next_value_tensor = torch.FloatTensor([next_value]).to(self.device)
            next_values = torch.cat([values[1:], next_value_tensor], 0)
        
        # Initialize advantages
        advantages = torch.zeros_like(rewards).to(self.device)
        
        # Calculate GAE
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
            
        return advantages
        
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
            ckpt_path = f"checkpoints/ppo_checkpoint_{env_name}_{suffix}"
            
        print(f'Saving models to {ckpt_path}')
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
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
            checkpoint = torch.load(ckpt_path)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if evaluate:
                self.actor_critic.eval()
            else:
                self.actor_critic.train()

class PPOKL(object):
    """
    Proximal Policy Optimization with KL Divergence penalty (PPOKL).
    PPOKL uses a penalty based on KL divergence 
    between old and new policy to constrain policy updates.
    """
    def __init__(self, num_inputs, action_space, args):
        """
        Initialize the PPOKL agent.
        
        Args:
            num_inputs (int): Dimension of state space
            action_space: Action space with shape and bounds
            args: Configuration arguments
        """
        # Algorithm parameters
        self.gamma = args.gamma                   # Discount factor
        self.gae_lambda = args.gae_lambda         # GAE parameter
        self.target_kl = args.target_kl           # Target KL divergence
        self.ppo_epochs = args.ppo_epochs         # Number of PPO epochs
        self.num_mini_batch = args.num_mini_batch # Number of minibatches for PPO
        self.value_loss_coef = args.value_loss_coef     # Value loss coefficient
        self.entropy_coef = args.entropy_coef           # Entropy coefficient
        self.max_grad_norm = args.max_grad_norm         # Max gradient norm for clipping
        self.use_clipped_value_loss = args.use_clipped_value_loss  # Whether to use clipped value loss
        
        # KL penalty coefficient and adaptation parameters
        self.kl_beta = args.initial_kl_beta       # Initial KL penalty coefficient
        self.kl_adapt_factor = args.kl_adapt_factor  # Factor for adapting KL coefficient
        self.min_kl_beta = args.min_kl_beta       # Minimum KL penalty coefficient
        self.max_kl_beta = args.max_kl_beta       # Maximum KL penalty coefficient
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic network
        self.actor_critic = ActorCritic(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.optimizer = Adam(self.actor_critic.parameters(), lr=args.lr)
        
    def select_action(self, state, evaluate=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: Current state
            evaluate (bool): Whether to evaluate (deterministic) or explore (stochastic)
            
        Returns:
            tuple: (action, value, log_prob, entropy)
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        # Get action, log probability, entropy and value
        action, log_prob, entropy, value = self.actor_critic.act(state, deterministic=evaluate)
        
        # Convert all tensors to numpy for storage in RolloutStorage
        action_np = action.detach().cpu().numpy()[0]
        value_np = value.detach().cpu().numpy()[0]
        log_prob_np = log_prob.detach().cpu().numpy()[0]
        entropy_np = entropy.detach().cpu().numpy()[0]
        
        return action_np, value_np, log_prob_np, entropy_np
    
    def update_parameters(self, rollouts):
        """
        Update policy and value parameters using collected rollouts with KL penalty.
        
        Args:
            rollouts: Collected trajectories (either a RolloutStorage object or a dictionary)
            
        Returns:
            dict: Loss values and metrics for logging
        """
        # Get data dict if rollouts is a RolloutStorage object
        if hasattr(rollouts, 'get_data'):
            rollouts = rollouts.get_data()
            
        # Calculate advantages if not already present in rollouts
        if 'advantages' not in rollouts:
            advantages = self._compute_advantages(rollouts)
        else:
            advantages = rollouts['advantages']
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_epoch = 0
        kl_epoch = 0
        
        # Perform multiple epochs of updates
        for _ in range(self.ppo_epochs):
            # Generate random permutation for mini-batches
            indices = np.random.permutation(len(rollouts['states']))
            batch_size = len(indices) // self.num_mini_batch
            
            epoch_kl = 0
            num_batches = 0
            
            # Process each mini-batch
            for start_idx in range(0, len(indices), batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                states_batch = torch.FloatTensor(rollouts['states'][batch_indices]).to(self.device)
                actions_batch = torch.FloatTensor(rollouts['actions'][batch_indices]).to(self.device)
                old_log_probs_batch = torch.FloatTensor(rollouts['log_probs'][batch_indices]).to(self.device)
                returns_batch = torch.FloatTensor(rollouts['returns'][batch_indices]).to(self.device)
                advantages_batch = advantages[batch_indices]
                
                # Forward pass to get new distribution
                mean, log_std, values = self.actor_critic(states_batch)
                new_dist = torch.distributions.Normal(mean, log_std.exp())
                
                # Get new action log probabilities
                new_log_probs = new_dist.log_prob(actions_batch).sum(-1, keepdim=True)
                entropy = new_dist.entropy().mean()
                
                # Compute ratio for PPO (importance sampling correction)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                
                # Compute surrogate objective (policy loss)
                surrogate = ratio * advantages_batch
                
                # Compute KL divergence between old and new policy
                # For Gaussian policy, KL can be computed analytically
                old_mean = mean.detach()
                old_std = log_std.exp().detach()
                
                # KL divergence for Gaussian distributions
                kl = torch.log(old_std/log_std.exp()) + (log_std.exp().pow(2) + (mean - old_mean).pow(2)) / (2 * old_std.pow(2)) - 0.5
                kl = kl.sum(-1, keepdim=True).mean()
                
                # Policy loss with KL penalty
                action_loss = -(surrogate - self.kl_beta * kl)
                action_loss = action_loss.mean()
                
                # Value loss
                if self.use_clipped_value_loss:
                    # Still use clipped value loss as in PPO-Clip for stable value updates
                    # Could also use a non-clipped version if preferred
                    old_values_batch = torch.FloatTensor(rollouts['values'][batch_indices]).to(self.device)
                    value_pred_clipped = old_values_batch + \
                                         (values - old_values_batch).clamp(-0.2, 0.2)  # Use same clip range as policy
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    # Standard value loss
                    value_loss = 0.5 * (returns_batch - values).pow(2).mean()
                
                # Overall loss
                loss = value_loss * self.value_loss_coef + action_loss - entropy * self.entropy_coef
                
                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
                # Record metrics
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_epoch += entropy.item()
                kl_epoch += kl.item()
                
                epoch_kl += kl.item()
                num_batches += 1
            
            # Adapt KL penalty coefficient after each epoch
            if num_batches > 0:
                avg_kl = epoch_kl / num_batches
                
                # Adjust beta (KL penalty coefficient) based on KL divergence
                if avg_kl < self.target_kl / 1.5:
                    self.kl_beta = max(self.min_kl_beta, self.kl_beta / self.kl_adapt_factor)
                elif avg_kl > self.target_kl * 1.5:
                    self.kl_beta = min(self.max_kl_beta, self.kl_beta * self.kl_adapt_factor)
        
        # Average metrics over epochs and mini-batches
        num_updates = self.ppo_epochs * (len(indices) // batch_size)
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_epoch /= num_updates
        kl_epoch /= num_updates
        
        return {
            'value_loss': value_loss_epoch,
            'action_loss': action_loss_epoch,
            'entropy': entropy_epoch,
            'kl': kl_epoch,
            'kl_beta': self.kl_beta
        }
    
    def _compute_advantages(self, rollouts, next_value=None):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rollouts: Collected trajectories
            next_value: Value estimate for the state after the last stored state.
                        If None, uses the last value in rollouts.
            
        Returns:
            torch.Tensor: Computed advantages
        """
        # Convert to tensors
        rewards = torch.FloatTensor(rollouts['rewards']).to(self.device)
        values = torch.FloatTensor(rollouts['values']).to(self.device)
        masks = torch.FloatTensor(rollouts['masks']).to(self.device)
        
        # For the last value, use provided next_value or the last value in rollouts
        if next_value is None:
            next_values = torch.cat([values[1:], values[-1:]], 0)
        else:
            next_value_tensor = torch.FloatTensor([next_value]).to(self.device)
            next_values = torch.cat([values[1:], next_value_tensor], 0)
        
        # Initialize advantages
        advantages = torch.zeros_like(rewards).to(self.device)
        
        # Calculate GAE
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
            
        return advantages
        
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
            ckpt_path = f"checkpoints/ppokl_checkpoint_{env_name}_{suffix}"
            
        print(f'Saving models to {ckpt_path}')
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
            checkpoint = torch.load(ckpt_path)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load KL beta coefficient if available
            if 'kl_beta' in checkpoint:
                self.kl_beta = checkpoint['kl_beta']

            if evaluate:
                self.actor_critic.eval()
            else:
                self.actor_critic.train()