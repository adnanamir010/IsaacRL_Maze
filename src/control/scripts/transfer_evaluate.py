#!/usr/bin/env python3

import os
import numpy as np
import torch
import argparse
import json
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback


# Initialize Isaac Sim if available
try:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    
    # Import Isaac Sim modules
    import omni
    from omni.isaac.core import World
    import rclpy
    
    # Import Isaac environment and utilities
    from isaac_environment import DDEnv
    from isaac_utils import ModelStateSubscriber, LidarSubscriber, CollisionDetector, evaluate_policy
    import global_vars
    
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("Warning: Isaac Sim dependencies not found. Only model transfer will be available, not evaluation.")

# Import both agent sets
from agents import SAC as GymSAC
from agents import PPOCLIP as GymPPOCLIP
from agents import PPOKL as GymPPOKL
from isaac_agents import SAC as IsaacSAC

def parse_arguments():
    """Parse command line arguments for the model transfer and evaluation script"""
    parser = argparse.ArgumentParser(description='Transfer and evaluate trained models from Gym to Isaac Sim')
    
    # Model source parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                     help='Path to the checkpoint file from gym environment')
    parser.add_argument('--algorithm', type=str, required=True, choices=["SAC", "PPOCLIP", "PPOKL"],
                     help='Algorithm type: SAC | PPOCLIP | PPOKL')
    
    # Model transfer parameters
    parser.add_argument('--output-dir', type=str, default='isaac_models',
                     help='Directory to save transferred models (default: isaac_models)')
    parser.add_argument('--skip-eval', action='store_true',
                     help='Skip evaluation in Isaac Sim (default: False)')
    
    # Parameters for agent reconstruction
    parser.add_argument('--state-dim', type=int, default=22,
                     help='State dimension (default: 22 for VectorizedDD)')
    parser.add_argument('--action-dim', type=int, default=1,
                     help='Action dimension (default: 1)')
    parser.add_argument('--hidden-size', type=int, default=256,
                     help='Hidden layer size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99,
                     help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                     help='Soft update coefficient (default: 0.005)')
    
    # SAC parameters
    parser.add_argument('--policy', default="Gaussian",
                     help='Policy Type for SAC: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--alpha', type=float, default=0.2,
                     help='Temperature parameter for SAC (default: 0.2)')
    parser.add_argument('--automatic-entropy-tuning', action='store_true', default=True,
                     help='Use automatic entropy tuning for SAC (default: True)')
    parser.add_argument('--use-twin-critic', action='store_true', default=True,
                     help='Use twin critic for SAC (default: True)')
    parser.add_argument('--entropy-mode', choices=['none', 'fixed', 'adaptive'], default='adaptive',
                     help='Entropy mode for SAC: none, fixed, or adaptive (default: adaptive)')
    
    # PPO parameters
    parser.add_argument('--clip-param', type=float, default=0.1,
                     help='PPO clip parameter (default: 0.1)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                     help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                     help='Entropy coefficient (default: 0.01)')
    parser.add_argument('--kl-target', type=float, default=0.005,
                     help='KL target for PPOKL (default: 0.005)')
    parser.add_argument('--kl-coef', type=float, default=1.0,
                     help='Initial KL coefficient for PPOKL (default: 1.0)')
    
    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=5,
                     help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--max-steps', type=int, default=220,
                     help='Maximum steps per episode (default: 220)')
    parser.add_argument('--seed', type=int, default=123456,
                     help='Random seed (default: 123456)')
    parser.add_argument('--verbose', action='store_true', default=True,
                     help='Print detailed information (default: True)')
    
    # System parameters
    parser.add_argument('--cuda', action='store_true', default=True,
                     help='Use CUDA if available (default: True)')
    
    return parser.parse_args()

def create_gym_agent(args):
    """
    Create an agent from the gym implementation for loading the checkpoint
    
    Args:
        args: Command line arguments
        
    Returns:
        agent: Initialized gym agent
    """
    # Configure device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    # Define a simple action space for agent creation
    class DummyActionSpace:
        def __init__(self):
            self.shape = [args.action_dim]
            self.high = np.array([1.0] * args.action_dim)
            self.low = np.array([-1.0] * args.action_dim)
            self.n = args.action_dim  # For discrete action spaces
    
    action_space = DummyActionSpace()
    
    # Create the appropriate agent based on algorithm
    if args.algorithm == "SAC":
        # For SAC, we need to handle different entropy modes
        class SACArgs:
            def __init__(self, args):
                self.gamma = args.gamma
                self.tau = args.tau
                self.lr = 0.0003  # Default SAC learning rate
                self.alpha = args.alpha
                self.policy = args.policy
                self.target_update_interval = 1  # Default value
                self.hidden_size = args.hidden_size
                self.automatic_entropy_tuning = args.automatic_entropy_tuning
                self.cuda = args.cuda
                self.use_twin_critic = args.use_twin_critic
                self.entropy_mode = args.entropy_mode
        
        sac_args = SACArgs(args)
        agent = GymSAC(args.state_dim, action_space, sac_args)
        print(f"Created Gym SAC agent with {'twin' if args.use_twin_critic else 'single'} critic")
    
    elif args.algorithm == "PPOCLIP":
        agent = GymPPOCLIP(args.state_dim, action_space, args)
        print(f"Created Gym PPOCLIP agent with clip parameter: {args.clip_param}")
    
    elif args.algorithm == "PPOKL":
        agent = GymPPOKL(args.state_dim, action_space, args)
        print(f"Created Gym PPOKL agent with KL target: {args.kl_target}")
    
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    return agent

def create_isaac_agent(args):
    """
    Create an agent from the Isaac implementation for transferring parameters
    
    Args:
        args: Command line arguments
        
    Returns:
        agent: Initialized Isaac agent
    """
    # Define a simple action space for agent creation
    class DummyActionSpace:
        def __init__(self):
            self.shape = [args.action_dim]
            self.high = np.array([1.0] * args.action_dim)
            self.low = np.array([-1.0] * args.action_dim)
    
    action_space = DummyActionSpace()
    
    # For now, only create SAC agent as the PPO variants might not be compatible
    if args.algorithm == "SAC":
        # For SAC, we need to handle different entropy modes
        class SACArgs:
            def __init__(self, args):
                self.gamma = args.gamma
                self.tau = args.tau
                self.lr = 0.0003  # Default SAC learning rate
                self.alpha = args.alpha
                self.policy = args.policy
                self.target_update_interval = 1  # Default value
                self.hidden_size = args.hidden_size
                self.automatic_entropy_tuning = args.automatic_entropy_tuning
                self.cuda = args.cuda
        
        sac_args = SACArgs(args)
        agent = IsaacSAC(args.state_dim, action_space, sac_args)
        print(f"Created Isaac SAC agent")
    
    elif args.algorithm == "PPOCLIP" or args.algorithm == "PPOKL":
        # For PPO variants, we'll skip the actual agent creation
        # since the architectures don't match well
        print(f"Note: {args.algorithm} agent won't be created in Isaac Sim due to architecture differences")
        print("The model parameters will be extracted and saved for manual integration")
        agent = None
    
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")
    
    return agent

def extract_model_parameters(gym_agent, args):
    """
    Extract model parameters from gym agent
    
    Args:
        gym_agent: The gym agent with loaded checkpoint
        args: Command line arguments
        
    Returns:
        dict: Model parameters for transfer
    """
    params = {
        'algorithm': args.algorithm,
        'state_dim': args.state_dim,
        'action_dim': args.action_dim,
        'hidden_size': args.hidden_size,
        'parameters': {}
    }
    
    if args.algorithm == "SAC":
        # Extract SAC parameters
        params['parameters']['policy'] = {}
        params['parameters']['critic'] = {}
        params['parameters']['critic_target'] = {}
        
        # Policy parameters
        for name, param in gym_agent.policy.named_parameters():
            params['parameters']['policy'][name] = param.clone().detach().cpu().numpy()
        
        # Critic parameters
        for name, param in gym_agent.critic.named_parameters():
            params['parameters']['critic'][name] = param.clone().detach().cpu().numpy()
        
        # Target critic parameters
        for name, param in gym_agent.critic_target.named_parameters():
            params['parameters']['critic_target'][name] = param.clone().detach().cpu().numpy()
        
        # Alpha parameters
        if hasattr(gym_agent, 'log_alpha'):
            params['parameters']['log_alpha'] = gym_agent.log_alpha.clone().detach().cpu().numpy()
            params['parameters']['alpha'] = gym_agent.alpha
        
        # Additional SAC parameters
        params['policy_type'] = args.policy
        params['use_twin_critic'] = args.use_twin_critic
        params['entropy_mode'] = args.entropy_mode
    
    elif args.algorithm == "PPOCLIP" or args.algorithm == "PPOKL":
        # Extract PPO parameters
        params['parameters']['policy'] = {}
        params['parameters']['value_net'] = {}
        
        # Policy parameters
        for name, param in gym_agent.policy.named_parameters():
            params['parameters']['policy'][name] = param.clone().detach().cpu().numpy()
        
        # Value network parameters
        for name, param in gym_agent.value_net.named_parameters():
            params['parameters']['value_net'][name] = param.clone().detach().cpu().numpy()
        
        # Algorithm-specific parameters
        if args.algorithm == "PPOCLIP":
            params['clip_param'] = args.clip_param
        else:  # PPOKL
            params['kl_target'] = args.kl_target
            params['kl_coef'] = args.kl_coef
            if hasattr(gym_agent, 'kl_beta'):
                params['kl_beta'] = float(gym_agent.kl_beta)
        
        params['value_loss_coef'] = args.value_loss_coef
        params['entropy_coef'] = args.entropy_coef
    
    # Print parameter shapes for debugging
    if args.verbose:
        print("\n===== Model Parameter Shapes =====")
        for section in params['parameters']:
            print(f"{section}:")
            if isinstance(params['parameters'][section], dict):
                for name, param in params['parameters'][section].items():
                    if isinstance(param, np.ndarray):
                        print(f"  {name}: {param.shape}")
            elif isinstance(params['parameters'][section], np.ndarray):
                print(f"  Shape: {params['parameters'][section].shape}")
        print("================================\n")
    
    return params

def transfer_sac_parameters(gym_agent, isaac_agent, args):
    """
    Transfer SAC parameters from gym model to Isaac model
    
    Args:
        gym_agent: The gym agent with loaded checkpoint
        isaac_agent: The Isaac agent to transfer parameters to
        args: Command line arguments
        
    Returns:
        bool: Success status
    """
    try:
        print("Transferring SAC parameters...")
        
        # 1. Transfer policy parameters
        gym_policy_dict = gym_agent.policy.state_dict()
        isaac_policy_dict = isaac_agent.policy.state_dict()
        
        # Create a new state dict for Isaac agent by mapping keys
        isaac_policy_new_dict = {}
        
        # Print keys for debugging
        if args.verbose:
            print("\nGym policy keys:")
            for key in gym_policy_dict.keys():
                print(f"  {key}: {gym_policy_dict[key].shape}")
            print("\nIsaac policy keys:")
            for key in isaac_policy_dict.keys():
                print(f"  {key}: {isaac_policy_dict[key].shape}")
        
        # Try to match parameters by name and shape
        for isaac_key, isaac_param in isaac_policy_dict.items():
            if isaac_key in gym_policy_dict and gym_policy_dict[isaac_key].shape == isaac_param.shape:
                # Direct match
                isaac_policy_new_dict[isaac_key] = gym_policy_dict[isaac_key]
            else:
                # Try to find a matching key by name or shape
                matching_key = None
                for gym_key, gym_param in gym_policy_dict.items():
                    # Check if shapes match and names are similar
                    if gym_param.shape == isaac_param.shape:
                        if gym_key.endswith(isaac_key.split('.')[-1]) or isaac_key.endswith(gym_key.split('.')[-1]):
                            matching_key = gym_key
                            break
                
                if matching_key:
                    isaac_policy_new_dict[isaac_key] = gym_policy_dict[matching_key]
                    if args.verbose:
                        print(f"Matched {isaac_key} with {matching_key}")
                else:
                    # If no match found, keep original parameter
                    isaac_policy_new_dict[isaac_key] = isaac_param
                    print(f"Warning: No match found for policy parameter {isaac_key}")
        
        # Load the new state dict
        isaac_agent.policy.load_state_dict(isaac_policy_new_dict)
        
        # 2. Transfer critic parameters using the same approach
        gym_critic_dict = gym_agent.critic.state_dict()
        isaac_critic_dict = isaac_agent.critic.state_dict()
        
        isaac_critic_new_dict = {}
        for isaac_key, isaac_param in isaac_critic_dict.items():
            if isaac_key in gym_critic_dict and gym_critic_dict[isaac_key].shape == isaac_param.shape:
                isaac_critic_new_dict[isaac_key] = gym_critic_dict[isaac_key]
            else:
                matching_key = None
                for gym_key, gym_param in gym_critic_dict.items():
                    if gym_param.shape == isaac_param.shape:
                        if gym_key.endswith(isaac_key.split('.')[-1]) or isaac_key.endswith(gym_key.split('.')[-1]):
                            matching_key = gym_key
                            break
                
                if matching_key:
                    isaac_critic_new_dict[isaac_key] = gym_critic_dict[matching_key]
                    if args.verbose:
                        print(f"Matched {isaac_key} with {matching_key}")
                else:
                    isaac_critic_new_dict[isaac_key] = isaac_param
                    print(f"Warning: No match found for critic parameter {isaac_key}")
        
        # Load the new state dict
        isaac_agent.critic.load_state_dict(isaac_critic_new_dict)
        
        # 3. Also transfer target critic parameters
        isaac_agent.critic_target.load_state_dict(isaac_critic_new_dict)
        
        # 4. Transfer alpha parameters if using automatic entropy tuning
        if args.automatic_entropy_tuning and hasattr(gym_agent, 'log_alpha') and hasattr(isaac_agent, 'log_alpha'):
            isaac_agent.log_alpha.data = gym_agent.log_alpha.data.clone()
            isaac_agent.alpha = gym_agent.alpha
        
        print("SAC parameters transferred successfully!")
        return True
    
    except Exception as e:
        print(f"Error transferring SAC parameters: {e}")
        traceback.print_exc()
        return False

def save_model_parameters(parameters, args, success=True):
    """
    Save the model parameters to files for later use
    
    Args:
        parameters: Dictionary of model parameters
        args: Command line arguments
        success: Whether the transfer was successful
        
    Returns:
        str: Path to the saved parameters
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_base = os.path.basename(args.checkpoint).replace(".", "_")
    param_name = f"isaac_{args.algorithm}_{checkpoint_base}_{timestamp}_params.npz"
    param_path = os.path.join(args.output_dir, param_name)
    
    # Save parameters to NPZ file
    try:
        # Extract parameters into numpy arrays
        np_params = {}
        
        # Convert parameters to numpy arrays
        for section in parameters['parameters']:
            if isinstance(parameters['parameters'][section], dict):
                for name, param in parameters['parameters'][section].items():
                    np_params[f"{section}_{name}"] = param
            else:
                np_params[section] = parameters['parameters'][section]
        
        # Save to NPZ file
        np.savez(param_path, **np_params)
        print(f"Model parameters saved to {param_path}")
        
        # Also save metadata to JSON
        info_name = f"isaac_{args.algorithm}_{checkpoint_base}_{timestamp}_info.json"
        info_path = os.path.join(args.output_dir, info_name)
        
        # Create metadata
        metadata = {k: v for k, v in parameters.items() if k != 'parameters'}
        metadata['original_checkpoint'] = args.checkpoint
        metadata['timestamp'] = timestamp
        metadata['success'] = success
        
        # Add parameter shapes to metadata
        metadata['parameter_shapes'] = {}
        for section in parameters['parameters']:
            if isinstance(parameters['parameters'][section], dict):
                metadata['parameter_shapes'][section] = {}
                for name, param in parameters['parameters'][section].items():
                    if isinstance(param, np.ndarray):
                        metadata['parameter_shapes'][section][name] = list(param.shape)
            elif isinstance(parameters['parameters'][section], np.ndarray):
                metadata['parameter_shapes'][section] = list(parameters['parameters'][section].shape)
        
        # Save metadata
        with open(info_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model metadata saved to {info_path}")
        
        # If SAC was successfully transferred to an Isaac agent,
        # also save the agent checkpoint
        if args.algorithm == "SAC" and success and 'isaac_agent' in locals():
            checkpoint_name = f"isaac_{args.algorithm}_{checkpoint_base}_{timestamp}.pt"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            isaac_agent.save_checkpoint("isaac", suffix=timestamp, ckpt_path=checkpoint_path)
            print(f"Isaac SAC agent saved to {checkpoint_path}")
        
        return param_path
    
    except Exception as e:
        print(f"Error saving model parameters: {e}")
        traceback.print_exc()
        return None

def create_parameter_usage_guide(parameters, args):
    """
    Create a guide for how to use the saved parameters
    
    Args:
        parameters: Dictionary of model parameters
        args: Command line arguments
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    guide_name = f"isaac_{args.algorithm}_{timestamp}_usage_guide.md"
    guide_path = os.path.join(args.output_dir, guide_name)
    
    with open(guide_path, 'w') as f:
        f.write(f"# Usage Guide for Transferred {args.algorithm} Model\n\n")
        f.write(f"This guide explains how to use the extracted model parameters from {args.checkpoint}.\n\n")
        
        f.write("## Model Information\n\n")
        f.write(f"- Algorithm: {args.algorithm}\n")
        f.write(f"- State Dimension: {args.state_dim}\n")
        f.write(f"- Action Dimension: {args.action_dim}\n")
        f.write(f"- Hidden Size: {args.hidden_size}\n")
        
        if args.algorithm == "SAC":
            f.write(f"- Policy Type: {args.policy}\n")
            f.write(f"- Entropy Mode: {args.entropy_mode}\n")
            f.write(f"- Twin Critic: {args.use_twin_critic}\n")
        elif args.algorithm == "PPOCLIP":
            f.write(f"- Clip Parameter: {args.clip_param}\n")
            f.write(f"- Value Loss Coefficient: {args.value_loss_coef}\n")
            f.write(f"- Entropy Coefficient: {args.entropy_coef}\n")
        elif args.algorithm == "PPOKL":
            f.write(f"- KL Target: {args.kl_target}\n")
            f.write(f"- KL Coefficient: {args.kl_coef}\n")
            f.write(f"- Value Loss Coefficient: {args.value_loss_coef}\n")
            f.write(f"- Entropy Coefficient: {args.entropy_coef}\n")
        
        f.write("\n## Parameter Files\n\n")
        f.write("The model parameters are saved in two files:\n\n")
        f.write("1. A `.npz` file containing the actual parameter values\n")
        f.write("2. A `.json` file containing metadata about the parameters\n\n")
        
        f.write("## Loading Parameters\n\n")
        
        if args.algorithm == "SAC":
            f.write("### For SAC\n\n")
            f.write("SAC parameters can be loaded directly into an Isaac SAC agent:\n\n")
            f.write("```python\n")
            f.write("import numpy as np\n")
            f.write("import torch\n")
            f.write("from isaac_agents import SAC\n\n")
            
            f.write("# Define action space\n")
            f.write("class ActionSpace:\n")
            f.write("    def __init__(self):\n")
            f.write("        self.shape = [1]  # Action dimension\n")
            f.write("        self.high = np.array([1.0])\n")
            f.write("        self.low = np.array([-1.0])\n\n")
            
            f.write("# Create agent\n")
            f.write("class SACArgs:\n")
            f.write("    def __init__(self):\n")
            f.write("        self.gamma = 0.99\n")
            f.write("        self.tau = 0.005\n")
            f.write("        self.lr = 0.0003\n")
            f.write(f"        self.alpha = {args.alpha}\n")
            f.write(f"        self.policy = '{args.policy}'\n")
            f.write("        self.target_update_interval = 1\n")
            f.write(f"        self.hidden_size = {args.hidden_size}\n")
            f.write(f"        self.automatic_entropy_tuning = {str(args.automatic_entropy_tuning).lower()}\n")
            f.write("        self.cuda = True\n\n")
            
            f.write("# Initialize agent\n")
            f.write("action_space = ActionSpace()\n")
            f.write(f"agent = SAC({args.state_dim}, action_space, SACArgs())\n\n")
            
            f.write("# Load parameters from NPZ file\n")
            f.write("params = np.load('path_to_params.npz')\n\n")
            
            f.write("# Create state dictionaries for each network\n")
            f.write("policy_dict = {}\n")
            f.write("critic_dict = {}\n\n")
            
            f.write("# Map parameters to the correct keys\n")
            f.write("for key in params.keys():\n")
            f.write("    if key.startswith('policy_'):\n")
            f.write("        # Extract parameter name without the 'policy_' prefix\n")
            f.write("        param_name = key[7:]\n")
            f.write("        policy_dict[param_name] = torch.tensor(params[key])\n")
            f.write("    elif key.startswith('critic_'):\n")
            f.write("        # Extract parameter name without the 'critic_' prefix\n")
            f.write("        param_name = key[7:]\n")
            f.write("        critic_dict[param_name] = torch.tensor(params[key])\n\n")
            
            f.write("# Load parameters into the agent\n")
            f.write("agent.policy.load_state_dict(policy_dict)\n")
            f.write("agent.critic.load_state_dict(critic_dict)\n")
            f.write("agent.critic_target.load_state_dict(critic_dict)\n\n")
            
            f.write("# If using automatic entropy tuning, also load alpha\n")
            f.write("if 'log_alpha' in params:\n")
            f.write("    agent.log_alpha.data = torch.tensor(params['log_alpha'])\n")
            f.write("    agent.alpha = float(params['alpha'])\n")
            f.write("```\n\n")
        
        elif args.algorithm == "PPOCLIP" or args.algorithm == "PPOKL":
            f.write("### For PPO Variants\n\n")
            f.write(f"The {args.algorithm} model architecture in `isaac_agents.py` differs significantly from the one in `agents.py`. ")
            f.write("You will need to create a custom agent class that matches the architecture in `isaac_agents.py` and then manually map the parameters.\n\n")
            
            f.write("Here's a basic outline of how to load the parameters:\n\n")
            
            f.write("```python\n")
            f.write("import numpy as np\n")
            f.write("import torch\n")
            f.write("import json\n\n")
            
            f.write("# Load parameters from NPZ file\n")
            f.write("params = np.load('path_to_params.npz')\n\n")
            
            f.write("# Load metadata\n")
            f.write("with open('path_to_info.json', 'r') as f:\n")
            f.write("    metadata = json.load(f)\n\n")
            
            f.write("# Extract policy and value network parameters\n")
            f.write("policy_params = {}\n")
            f.write("value_params = {}\n\n")
            
            f.write("for key in params.keys():\n")
            f.write("    if key.startswith('policy_'):\n")
            f.write("        param_name = key[7:]\n")
            f.write("        policy_params[param_name] = torch.tensor(params[key])\n")
            f.write("    elif key.startswith('value_net_'):\n")
            f.write("        param_name = key[10:]\n")
            f.write("        value_params[param_name] = torch.tensor(params[key])\n\n")
            
            f.write("# Now you need to create a custom implementation of ActorCritic\n")
            f.write("# that matches the architecture in isaac_agents.py\n")
            f.write("# and manually map the parameters from policy_params and value_params\n")
            f.write("```\n\n")
            
            f.write("## Implementing Custom PPO Agent\n\n")
            f.write("For PPO variants, you'll need to create a custom agent class in Isaac Sim ")
            f.write("that matches the architecture of the provided parameters. ")
            f.write("The main challenge is that the gym implementation uses separate policy and value networks, ")
            f.write("while the Isaac implementation uses a combined actor-critic network.\n\n")
            
            f.write("### Parameter Mapping\n\n")
            f.write("Here's a guide for mapping the parameters:\n\n")
            
            f.write("1. **Policy Network to Actor**:\n")
            f.write("   - The policy network parameters should be mapped to the actor part of the actor-critic network\n")
            f.write("   - For example, `policy.linear1.weight` -> `actor_critic.actor.linear1.weight`\n\n")
            
            f.write("2. **Value Network to Critic**:\n")
            f.write("   - The value network parameters should be mapped to the critic part of the actor-critic network\n")
            f.write("   - For example, `value_net.linear1.weight` -> `actor_critic.critic.linear1.weight`\n\n")
            
            f.write("3. **Creating Custom Agent**:\n")
            f.write("   - You may need to modify the `ActorCritic` class in `isaac_agents.py` to match the architecture\n")
            f.write("   - Alternatively, you can create a new agent class that uses the extracted parameters\n\n")
        
        f.write("## Testing the Transferred Model\n\n")
        f.write("After loading the parameters, you can test the agent in the Isaac Sim environment:\n\n")
        
        f.write("```python\n")
        f.write("# Initialize Isaac Sim\n")
        f.write("from isaacsim import SimulationApp\n")
        f.write("simulation_app = SimulationApp({\"headless\": False})\n\n")
        
        f.write("# Import required modules\n")
        f.write("import omni\n")
        f.write("from omni.isaac.core import World\n")
        f.write("import rclpy\n\n")
        
        f.write("# Initialize ROS2\n")
        f.write("rclpy.init(args=None)\n\n")
        
        f.write("# Create environment\n")
        f.write("# ... (environment setup code)\n\n")
        
        f.write("# Run the agent\n")
        f.write("state = env.reset()\n")
        f.write("done = False\n")
        f.write("while not done:\n")
        f.write("    action = agent.select_action(state, evaluate=True)\n")
        f.write("    next_state, reward, done = env.step(action, step_count, max_steps)\n")
        f.write("    state = next_state\n")
        f.write("```\n\n")
        
        f.write("## Additional Notes\n\n")
        f.write("- The parameter shapes and names are available in the metadata JSON file\n")
        f.write("- You may need to adjust the network architecture to match the parameter shapes\n")
        f.write("- For PPO variants, you'll need to implement a custom approach due to architecture differences\n")
    
    print(f"Usage guide saved to {guide_path}")

def main():
    """Main function for transferring and evaluating models"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    
    print("\n===== Starting Model Parameter Extraction =====")
    print(f"Algorithm: {args.algorithm}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print("============================================\n")
    
    # Create gym agent
    gym_agent = create_gym_agent(args)
    
    # Load checkpoint
    print(f"Loading gym checkpoint from {args.checkpoint}")
    try:
        gym_agent.load_checkpoint(args.checkpoint)
        print("Gym checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading gym checkpoint: {e}")
        traceback.print_exc()
        return
    
    # Extract model parameters
    print("\nExtracting model parameters...")
    parameters = extract_model_parameters(gym_agent, args)
    
    # For SAC, we can try to transfer directly
    isaac_agent = None
    transfer_success = False
    
    if args.algorithm == "SAC":
        # Create Isaac agent
        isaac_agent = create_isaac_agent(args)
        
        # Transfer parameters
        transfer_success = transfer_sac_parameters(gym_agent, isaac_agent, args)
    else:
        # For PPO variants, we'll just extract parameters
        print("\nExtracted parameters for PPO variant - manual integration required")
        print("Creating usage guide for manual implementation...")
    
    # Save parameters
    param_path = save_model_parameters(parameters, args, transfer_success)
    
    # Create usage guide
    create_parameter_usage_guide(parameters, args)
    
    print("\n===== Model Parameter Extraction Complete =====")
    print(f"Original checkpoint: {args.checkpoint}")
    if param_path:
        print(f"Parameters saved to: {param_path}")
    if args.algorithm == "SAC" and transfer_success:
        print("SAC parameters were successfully transferred")
    else:
        print("Parameters extracted - see usage guide for integration instructions")
    print("==============================================\n")

if __name__ == "__main__":
    main()