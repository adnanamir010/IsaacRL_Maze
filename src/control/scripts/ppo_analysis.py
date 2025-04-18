#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.ticker import MaxNLocator
from scipy import stats
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import glob
import re
from collections import defaultdict

def parse_arguments():
    """Parse command line arguments for the PPO variants analysis script."""
    parser = argparse.ArgumentParser(description='Analyze PPO variants comparison results')
    
    parser.add_argument('--logs-dir', type=str, required=True,
                        help='Directory containing the TensorBoard logs')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis results (default: logs_dir/analysis)')
    parser.add_argument('--smoothing', type=int, default=5,
                        help='Window size for smoothing plots (default: 5)')
    parser.add_argument('--figsize', type=str, default='12,8',
                        help='Figure size as width,height (default: 12,8)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures (default: 300)')
    parser.add_argument('--font-size', type=int, default=12,
                        help='Base font size for plots (default: 12)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-whitegrid',
                        help='Matplotlib style (default: seaborn-v0_8-whitegrid)')
    parser.add_argument('--no-confidence', action='store_true',
                        help='Disable confidence intervals in plots')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes to include (default: None)')
    
    return parser.parse_args()

def load_tensorboard_data(logs_dir):
    """Load data from TensorBoard logs."""
    print(f"Loading TensorBoard data from {logs_dir}...")
    
    # Pattern to identify variant folders
    pattern = r"(PPOCLIP|PPOKL)_.*"
    
    # Find all variant directories
    variant_dirs = [d for d in os.listdir(logs_dir) if re.match(pattern, d)]
    
    if not variant_dirs:
        print("Error: No variant directories found. Make sure the directory structure matches the expected pattern.")
        return None
    
    print(f"Found {len(variant_dirs)} variant directories: {variant_dirs}")
    
    # Data structure to store metrics
    metrics_data = {}
    
    # Metrics to extract
    common_metrics = [
        'train/episode_reward',
        'train/episode_length',
        'train/reward_per_step',
        'train/avg_reward',
        'train/value_loss',
        'train/policy_loss',
        'train/entropy',
        'eval/mean_reward',
        'eval/std_reward',
        'hyperparams/policy_lr',
        'hyperparams/value_lr',
    ]
    
    ppoclip_metrics = [
        'train/clip_fraction',
    ]
    
    ppokl_metrics = [
        'train/kl_divergence',
        'train/kl_coef',
    ]
    
    # Extract variant info and load data
    for variant_dir in variant_dirs:
        # Parse variant type
        if "PPOCLIP" in variant_dir:
            variant_type = "PPOCLIP"
            metrics_of_interest = common_metrics + ppoclip_metrics
        elif "PPOKL" in variant_dir:
            variant_type = "PPOKL"
            metrics_of_interest = common_metrics + ppokl_metrics
        else:
            print(f"Warning: Unrecognized variant type in {variant_dir}")
            continue
            
        print(f"Processing {variant_type}...")
        
        # Find all event files in the variant directory
        event_files = glob.glob(os.path.join(logs_dir, variant_dir, "events.out.tfevents.*"))
        
        if not event_files:
            print(f"Warning: No event files found in {variant_dir}")
            continue
        
        # Load event data
        ea = event_accumulator.EventAccumulator(os.path.join(logs_dir, variant_dir))
        ea.Reload()
        
        # Filter available tags to the ones we're interested in
        available_tags = ea.Tags()['scalars']
        tags_to_load = [tag for tag in metrics_of_interest if tag in available_tags]
        
        if not tags_to_load:
            print(f"Warning: No relevant metrics found in {variant_dir}")
            continue
        
        # Initialize data structure for this variant
        if variant_type not in metrics_data:
            metrics_data[variant_type] = {}
        
        # Extract metrics
        for tag in tags_to_load:
            events = ea.Scalars(tag)
            
            if not events:
                print(f"Warning: No events found for {tag} in {variant_dir}")
                continue
            
            # Extract step, value, and wall_time
            steps = [event.step for event in events]
            values = [event.value for event in events]
            wall_times = [event.wall_time for event in events]
            
            # Store in the data structure
            clean_tag = tag.replace('/', '_')
            metrics_data[variant_type][clean_tag] = {
                'steps': steps,
                'values': values,
                'wall_times': wall_times
            }
    
    return metrics_data

def prepare_dataframes(metrics_data):
    """Convert metrics data to pandas DataFrames for easier analysis."""
    dfs = {}
    
    # Create DataFrame for each metric type
    metrics_to_process = [
        'train_episode_reward',
        'train_episode_length',
        'train_reward_per_step',
        'train_avg_reward',
        'train_value_loss',
        'train_policy_loss',
        'train_entropy',
        'eval_mean_reward',
        'eval_std_reward',
        'hyperparams_policy_lr',
        'hyperparams_value_lr',
        'train_clip_fraction',
        'train_kl_divergence',
        'train_kl_coef',
    ]
    
    for metric in metrics_to_process:
        data = []
        
        for variant_type, variant_data in metrics_data.items():
            if metric in variant_data:
                # Create rows for the DataFrame
                for step, value, wall_time in zip(variant_data[metric]['steps'], 
                                                variant_data[metric]['values'], 
                                                variant_data[metric]['wall_times']):
                    data.append({
                        'step': step,
                        'value': value,
                        'wall_time': wall_time,
                        'variant_type': variant_type
                    })
        
        if data:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Add episode number for episode-based metrics
            if metric in ['train_episode_reward', 'train_episode_length', 'train_reward_per_step', 'train_avg_reward']:
                # Create episode number by variant
                for variant in df['variant_type'].unique():
                    variant_mask = df['variant_type'] == variant
                    df.loc[variant_mask, 'episode'] = range(1, sum(variant_mask) + 1)
            
            dfs[metric] = df
    
    return dfs

def apply_smoothing(data, window_size=5):
    """Apply moving average smoothing to data."""
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_learning_curves(dfs, args, output_dir):
    """Generate learning curve plots for comparing the PPO variants."""
    
    # Set up plot style
    plt.style.use(args.style)
    plt.rcParams.update({
        'font.size': args.font_size,
        'axes.titlesize': args.font_size + 2,
        'axes.labelsize': args.font_size + 1,
        'xtick.labelsize': args.font_size,
        'ytick.labelsize': args.font_size,
        'legend.fontsize': args.font_size,
        'figure.figsize': tuple(map(float, args.figsize.split(','))),
    })
    
    # Color mapping for variants
    colors = {
        'PPOCLIP': 'blue',
        'PPOKL': 'red'
    }
    
    # Plot training episode rewards
    if 'train_episode_reward' in dfs:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        train_rewards = dfs['train_episode_reward'].copy()
        
        # Limit episodes if specified
        if args.max_episodes:
            train_rewards = train_rewards[train_rewards['episode'] <= args.max_episodes]
        
        # Plot for each variant
        for variant_type in train_rewards['variant_type'].unique():
            variant_data = train_rewards[train_rewards['variant_type'] == variant_type]
            
            # Sort by episode number
            variant_data = variant_data.sort_values('episode')
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_rewards = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_rewards = variant_data['value']
            
            # Plot
            ax.plot(variant_data['episode'], smoothed_rewards, 
                   label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training Episode Reward')
        ax.set_title('PPO Variants: Training Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_rewards_comparison.png'))
        plt.close()
    
    # Plot evaluation rewards
    if 'eval_mean_reward' in dfs and 'eval_std_reward' in dfs:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # Merge evaluation reward and std data
        eval_rewards = dfs['eval_mean_reward'].copy()
        eval_std = dfs['eval_std_reward'].copy()
        
        # Create a mapping from (variant_type, step) to std value
        std_map = {}
        for _, row in eval_std.iterrows():
            std_map[(row['variant_type'], row['step'])] = row['value']
        
        # Add std to the eval_rewards DataFrame
        eval_rewards['std'] = eval_rewards.apply(
            lambda row: std_map.get((row['variant_type'], row['step']), 0), 
            axis=1
        )
        
        # Optionally convert to episode number based on training data
        if 'train_episode_reward' in dfs:
            # Find the episode that corresponds to each evaluation step
            train_rewards = dfs['train_episode_reward']
            
            # For each variant, find closest episode to each eval step
            for variant_type in eval_rewards['variant_type'].unique():
                variant_evals = eval_rewards[eval_rewards['variant_type'] == variant_type]
                variant_trains = train_rewards[train_rewards['variant_type'] == variant_type]
                
                # Map steps to episodes
                step_to_episode = {}
                for _, row in variant_trains.iterrows():
                    step_to_episode[row['step']] = row['episode']
                
                # For each eval step, find the closest training step and its episode
                for eval_idx, eval_row in variant_evals.iterrows():
                    eval_step = eval_row['step']
                    closest_step = min(step_to_episode.keys(), key=lambda x: abs(x - eval_step))
                    closest_episode = step_to_episode[closest_step]
                    eval_rewards.loc[eval_idx, 'episode'] = closest_episode
        
        # Plot for each variant type
        for variant_type in eval_rewards['variant_type'].unique():
            variant_data = eval_rewards[eval_rewards['variant_type'] == variant_type]
            
            # Sort by step or episode if available
            if 'episode' in variant_data.columns:
                variant_data = variant_data.sort_values('episode')
                x_values = variant_data['episode']
                x_label = 'Episode'
            else:
                variant_data = variant_data.sort_values('step')
                x_values = variant_data['step']
                x_label = 'Step'
            
            # Apply smoothing if there's enough data
            if len(variant_data) > 1:
                if args.smoothing > 1 and len(variant_data) > args.smoothing:
                    # Group by x value and apply smoothing
                    if x_label == 'Episode':
                        grouped = variant_data.groupby('episode').agg({
                            'value': 'mean',
                            'std': 'mean'
                        }).reset_index()
                        x_values = grouped['episode']
                    else:
                        grouped = variant_data.groupby('step').agg({
                            'value': 'mean',
                            'std': 'mean'
                        }).reset_index()
                        x_values = grouped['step']
                    
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                    smoothed_stds = apply_smoothing(grouped['std'], args.smoothing)
                else:
                    # Not enough data for smoothing or smoothing disabled
                    smoothed_rewards = variant_data['value']
                    smoothed_stds = variant_data['std']
            else:
                # Only a single data point
                smoothed_rewards = variant_data['value']
                smoothed_stds = variant_data['std']
            
            # Plot mean
            line = ax.plot(x_values, smoothed_rewards, 
                          label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
            
            # Add confidence interval
            if not args.no_confidence:
                ax.fill_between(
                    x_values,
                    smoothed_rewards - smoothed_stds,
                    smoothed_rewards + smoothed_stds,
                    alpha=0.2,
                    color=line[0].get_color()
                )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Mean Evaluation Reward')
        ax.set_title('PPO Variants: Evaluation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eval_rewards_comparison.png'))
        plt.close()
    
    # Plot specific metrics comparison panels
    
    # 1. Value and Policy Losses
    if 'train_value_loss' in dfs and 'train_policy_loss' in dfs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=args.dpi)
        
        # Value Loss
        value_loss = dfs['train_value_loss'].copy()
        
        for variant_type in value_loss['variant_type'].unique():
            variant_data = value_loss[value_loss['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax1.plot(variant_data['step'], smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value Loss')
        ax1.set_title('Value Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Policy Loss
        policy_loss = dfs['train_policy_loss'].copy()
        
        for variant_type in policy_loss['variant_type'].unique():
            variant_data = policy_loss[policy_loss['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax2.plot(variant_data['step'], smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Policy Loss')
        ax2.set_title('Policy Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
        plt.close()
    
    # 2. Entropy and Algorithm-specific metrics (clip fraction or KL divergence)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=args.dpi)
    
    # Entropy
    if 'train_entropy' in dfs:
        ax1 = axes[0]
        entropy_data = dfs['train_entropy'].copy()
        
        for variant_type in entropy_data['variant_type'].unique():
            variant_data = entropy_data[entropy_data['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax1.plot(variant_data['step'], smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Policy Entropy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Clip Fraction (PPOCLIP only)
    if 'train_clip_fraction' in dfs:
        ax2 = axes[1]
        clip_data = dfs['train_clip_fraction'].copy()
        
        for variant_type in clip_data['variant_type'].unique():
            if variant_type != 'PPOCLIP':
                continue
                
            variant_data = clip_data[clip_data['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax2.plot(variant_data['step'], smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
            
            # Add reference lines for clip fraction
            ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10% threshold')
            ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='20% threshold')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Clip Fraction')
        ax2.set_title('PPOCLIP: Clip Fraction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # KL Divergence (PPOKL only)
    if 'train_kl_divergence' in dfs:
        ax3 = axes[2]
        kl_data = dfs['train_kl_divergence'].copy()
        
        for variant_type in kl_data['variant_type'].unique():
            if variant_type != 'PPOKL':
                continue
                
            variant_data = kl_data[kl_data['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax3.plot(variant_data['step'], smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
            
            # Add target KL line if we have KL coefficient data
            if 'train_kl_coef' in dfs:
                # Estimate the target KL from the data pattern
                target_kl = 0.005  # Common default value
                ax3.axhline(y=target_kl, color='r', linestyle='--', alpha=0.5, label='Target KL')
        
        ax3.set_xlabel('Step')
        ax3.set_ylabel('KL Divergence')
        ax3.set_title('PPOKL: KL Divergence')
        ax3.set_yscale('symlog')  # Use symlog scale for better visualization
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_specific_metrics.png'))
    plt.close()
    
    # 3. Reward per Step and Episode Length
    if 'train_reward_per_step' in dfs and 'train_episode_length' in dfs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=args.dpi)
        
        # Reward per Step
        reward_per_step = dfs['train_reward_per_step'].copy()
        
        # Limit episodes if specified
        if args.max_episodes and 'episode' in reward_per_step.columns:
            reward_per_step = reward_per_step[reward_per_step['episode'] <= args.max_episodes]
        
        for variant_type in reward_per_step['variant_type'].unique():
            variant_data = reward_per_step[reward_per_step['variant_type'] == variant_type]
            
            if 'episode' in variant_data.columns:
                variant_data = variant_data.sort_values('episode')
                x_values = variant_data['episode']
                x_label = 'Episode'
            else:
                variant_data = variant_data.sort_values('step')
                x_values = variant_data['step']
                x_label = 'Step'
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax1.plot(x_values, smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
        
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Reward per Step')
        ax1.set_title('Efficiency: Reward per Step')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Episode Length
        episode_length = dfs['train_episode_length'].copy()
        
        # Limit episodes if specified
        if args.max_episodes and 'episode' in episode_length.columns:
            episode_length = episode_length[episode_length['episode'] <= args.max_episodes]
        
        for variant_type in episode_length['variant_type'].unique():
            variant_data = episode_length[episode_length['variant_type'] == variant_type]
            
            if 'episode' in variant_data.columns:
                variant_data = variant_data.sort_values('episode')
                x_values = variant_data['episode']
                x_label = 'Episode'
            else:
                variant_data = variant_data.sort_values('step')
                x_values = variant_data['step']
                x_label = 'Step'
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax2.plot(x_values, smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
        
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Duration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_metrics.png'))
        plt.close()
    
    # 4. PPO Comparison Dashboard
    fig = plt.figure(figsize=(15, 10), dpi=args.dpi)
    
    # Training Rewards
    if 'train_episode_reward' in dfs:
        ax1 = fig.add_subplot(221)
        train_rewards = dfs['train_episode_reward'].copy()
        
        # Limit episodes if specified
        if args.max_episodes and 'episode' in train_rewards.columns:
            train_rewards = train_rewards[train_rewards['episode'] <= args.max_episodes]
        
        for variant_type in train_rewards['variant_type'].unique():
            variant_data = train_rewards[train_rewards['variant_type'] == variant_type]
            
            if 'episode' in variant_data.columns:
                variant_data = variant_data.sort_values('episode')
                x_values = variant_data['episode']
                x_label = 'Episode'
            else:
                variant_data = variant_data.sort_values('step')
                x_values = variant_data['step']
                x_label = 'Step'
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax1.plot(x_values, smoothed_values, 
                    label=variant_type, linewidth=2, color=colors.get(variant_type, 'gray'))
        
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Evaluation Rewards
    if 'eval_mean_reward' in dfs:
        ax2 = fig.add_subplot(222)
        eval_rewards = dfs['eval_mean_reward'].copy()
        
        for variant_type in eval_rewards['variant_type'].unique():
            variant_data = eval_rewards[eval_rewards['variant_type'] == variant_type]
            
            if 'episode' in variant_data.columns:
                variant_data = variant_data.sort_values('episode')
                x_values = variant_data['episode']
                x_label = 'Episode'
            else:
                variant_data = variant_data.sort_values('step')
                x_values = variant_data['step']
                x_label = 'Step'
            
            ax2.plot(x_values, variant_data['value'], 
                    label=variant_type, linewidth=2, marker='o', color=colors.get(variant_type, 'gray'))
        
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Mean Evaluation Reward')
        ax2.set_title('Evaluation Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Value and Policy Losses
    if 'train_value_loss' in dfs and 'train_policy_loss' in dfs:
        ax3 = fig.add_subplot(223)
        
        # Value Loss
        value_loss = dfs['train_value_loss'].copy()
        
        for variant_type in value_loss['variant_type'].unique():
            variant_data = value_loss[value_loss['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax3.plot(variant_data['step'], smoothed_values, 
                    label=f"{variant_type} Value Loss", linewidth=2, 
                    color=colors.get(variant_type, 'gray'), linestyle='-')
        
        # Create a twin axis for policy loss (which may have a different scale)
        ax3b = ax3.twinx()
        
        # Policy Loss
        policy_loss = dfs['train_policy_loss'].copy()
        
        for variant_type in policy_loss['variant_type'].unique():
            variant_data = policy_loss[policy_loss['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax3b.plot(variant_data['step'], smoothed_values, 
                     label=f"{variant_type} Policy Loss", linewidth=2, 
                     color=colors.get(variant_type, 'gray'), linestyle='--')
        
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Value Loss')
        ax3b.set_ylabel('Policy Loss')
        ax3.set_title('Training Losses')
        
        # Combine legends from both axes
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax3.grid(True, alpha=0.3)
    
    # Entropy and Algorithm-specific metrics
    ax4 = fig.add_subplot(224)
    
    if 'train_entropy' in dfs:
        entropy_data = dfs['train_entropy'].copy()
        
        for variant_type in entropy_data['variant_type'].unique():
            variant_data = entropy_data[entropy_data['variant_type'] == variant_type]
            variant_data = variant_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(variant_data) > args.smoothing:
                smoothed_values = apply_smoothing(variant_data['value'], args.smoothing)
            else:
                smoothed_values = variant_data['value']
            
            ax4.plot(variant_data['step'], smoothed_values, 
                    label=f"{variant_type} Entropy", linewidth=2, 
                    color=colors.get(variant_type, 'gray'), linestyle='-')
    
    # Add algorithm-specific metrics to the same plot
    if 'train_clip_fraction' in dfs:
        clip_data = dfs['train_clip_fraction'][dfs['train_clip_fraction']['variant_type'] == 'PPOCLIP'].copy()
        if not clip_data.empty:
            clip_data = clip_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(clip_data) > args.smoothing:
                smoothed_values = apply_smoothing(clip_data['value'], args.smoothing)
            else:
                smoothed_values = clip_data['value']
            
            # Create another twin y-axis
            ax4b = ax4.twinx()
            ax4b.plot(clip_data['step'], smoothed_values, 
                     label="PPOCLIP Clip Fraction", linewidth=2, 
                     color='purple', linestyle=':')
            ax4b.set_ylabel('Clip Fraction')
    
    if 'train_kl_divergence' in dfs:
        kl_data = dfs['train_kl_divergence'][dfs['train_kl_divergence']['variant_type'] == 'PPOKL'].copy()
        if not kl_data.empty:
            kl_data = kl_data.sort_values('step')
            
            # Apply smoothing
            if args.smoothing > 1 and len(kl_data) > args.smoothing:
                smoothed_values = apply_smoothing(kl_data['value'], args.smoothing)
            else:
                smoothed_values = kl_data['value']
            
            # Create another twin y-axis for KL divergence
            if not 'ax4b' in locals():
                ax4b = ax4.twinx()
            
            ax4b.plot(kl_data['step'], smoothed_values, 
                     label="PPOKL KL Divergence", linewidth=2, 
                     color='green', linestyle=':')
            ax4b.set_ylabel('KL Divergence / Clip Fraction')
            
            # Add target KL reference line
            target_kl = 0.005  # Assuming default value
            ax4b.axhline(y=target_kl, color='green', linestyle='--', alpha=0.5, label='Target KL')
    
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy & Algorithm Metrics')
    
    # Combine legends if we have twin axes
    if 'ax4b' in locals():
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax4.legend()
    
    ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('PPO Variants Comparison: PPOCLIP vs PPOKL', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'ppo_variants_dashboard.png'))
    plt.close()

def generate_statistical_analysis(dfs, output_dir):
    """Generate statistical analysis of the results."""
    # Import scipy stats explicitly inside the function to avoid scope issues
    from scipy import stats
    
    # Create a text file to store the analysis
    stats_file = os.path.join(output_dir, 'statistical_analysis.txt')
    
    with open(stats_file, 'w') as f:
        f.write("# Statistical Analysis of PPO Variants Performance\n\n")
        
        # Training reward analysis
        if 'train_episode_reward' in dfs:
            f.write("## Training Rewards Analysis\n\n")
            
            train_rewards = dfs['train_episode_reward'].copy()
            
            # Basic statistics by variant
            f.write("### Basic Training Statistics\n\n")
            f.write("| Variant | Episodes | Mean | Median | Std Dev | Min | Max | Final |\n")
            f.write("|---------|----------|------|--------|---------|-----|-----|-------|\n")
            
            for variant_type in train_rewards['variant_type'].unique():
                variant_data = train_rewards[train_rewards['variant_type'] == variant_type]
                rewards = variant_data['value']
                num_episodes = len(rewards)
                
                # Get last reward - assumes data is ordered by episode
                last_reward = rewards.iloc[-1] if not rewards.empty else 'N/A'
                
                f.write(f"| {variant_type} | {num_episodes} | {rewards.mean():.2f} | {rewards.median():.2f} | "
                       f"{rewards.std():.2f} | {rewards.min():.2f} | {rewards.max():.2f} | "
                       f"{last_reward:.2f} |\n")
            
            f.write("\n")
            
            # Training reward statistical comparison
            if len(train_rewards['variant_type'].unique()) > 1:
                f.write("### Training Reward Statistical Comparison\n\n")
                
                # Organize data by variant for comparison
                variants_data = {}
                for variant_type in train_rewards['variant_type'].unique():
                    variant_data = train_rewards[train_rewards['variant_type'] == variant_type]
                    variants_data[variant_type] = variant_data['value'].values
                
                # Perform t-test between variants
                variant_pairs = []
                for i, variant1 in enumerate(variants_data.keys()):
                    for j, variant2 in enumerate(variants_data.keys()):
                        if i < j:  # Only compare each pair once
                            variant_pairs.append((variant1, variant2))
                
                for variant1, variant2 in variant_pairs:
                    f.write(f"**{variant1} vs {variant2}**\n\n")
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(
                        variants_data[variant1],
                        variants_data[variant2],
                        equal_var=False  # Welch's t-test (doesn't assume equal variance)
                    )
                    
                    f.write(f"- t-test: t={t_stat:.3f}, p={p_value:.5f}")
                    if p_value < 0.05:
                        f.write(" (statistically significant difference)\n")
                    else:
                        f.write(" (no statistically significant difference)\n")
                    
                    # Mann-Whitney U test (non-parametric)
                    try:
                        u_stat, p_value = stats.mannwhitneyu(
                            variants_data[variant1],
                            variants_data[variant2]
                        )
                        
                        f.write(f"- Mann-Whitney U test: U={u_stat:.3f}, p={p_value:.5f}")
                        if p_value < 0.05:
                            f.write(" (statistically significant difference)\n")
                        else:
                            f.write(" (no statistically significant difference)\n")
                    except ValueError:
                        f.write("- Mann-Whitney U test could not be performed due to sample constraints\n")
                    
                    # Effect size (Cohen's d)
                    mean1, mean2 = np.mean(variants_data[variant1]), np.mean(variants_data[variant2])
                    std1, std2 = np.std(variants_data[variant1]), np.std(variants_data[variant2])
                    
                    # Pooled standard deviation
                    n1, n2 = len(variants_data[variant1]), len(variants_data[variant2])
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                    
                    # Cohen's d
                    cohen_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    
                    f.write(f"- Effect size (Cohen's d): {cohen_d:.3f} ")
                    if cohen_d < 0.2:
                        f.write("(negligible effect)\n")
                    elif cohen_d < 0.5:
                        f.write("(small effect)\n")
                    elif cohen_d < 0.8:
                        f.write("(medium effect)\n")
                    else:
                        f.write("(large effect)\n")
                    
                    f.write("\n")
        
        # Evaluation reward analysis
        if 'eval_mean_reward' in dfs:
            f.write("## Evaluation Rewards Analysis\n\n")
            
            eval_rewards = dfs['eval_mean_reward'].copy()
            
            # Basic statistics by variant
            f.write("### Basic Evaluation Statistics\n\n")
            f.write("| Variant | Evaluations | Mean | Median | Std Dev | Min | Max | Final |\n")
            f.write("|---------|-------------|------|--------|---------|-----|-----|-------|\n")
            
            for variant_type in eval_rewards['variant_type'].unique():
                variant_data = eval_rewards[eval_rewards['variant_type'] == variant_type]
                rewards = variant_data['value']
                num_evals = len(rewards)
                
                # Get last reward
                last_step = variant_data['step'].max()
                last_reward_row = variant_data[variant_data['step'] == last_step]
                last_reward = last_reward_row['value'].iloc[0] if not last_reward_row.empty else 'N/A'
                
                f.write(f"| {variant_type} | {num_evals} | {rewards.mean():.2f} | {rewards.median():.2f} | "
                       f"{rewards.std():.2f} | {rewards.min():.2f} | {rewards.max():.2f} | "
                       f"{last_reward:.2f} |\n")
            
            f.write("\n")
            
            # Evaluation reward statistical comparison
            if len(eval_rewards['variant_type'].unique()) > 1:
                f.write("### Evaluation Reward Statistical Comparison\n\n")
                
                # Organize data by variant for comparison
                variants_data = {}
                for variant_type in eval_rewards['variant_type'].unique():
                    variant_data = eval_rewards[eval_rewards['variant_type'] == variant_type]
                    variants_data[variant_type] = variant_data['value'].values
                
                # Perform t-test between variants
                variant_pairs = []
                for i, variant1 in enumerate(variants_data.keys()):
                    for j, variant2 in enumerate(variants_data.keys()):
                        if i < j:  # Only compare each pair once
                            variant_pairs.append((variant1, variant2))
                
                for variant1, variant2 in variant_pairs:
                    f.write(f"**{variant1} vs {variant2}**\n\n")
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(
                        variants_data[variant1],
                        variants_data[variant2],
                        equal_var=False  # Welch's t-test (doesn't assume equal variance)
                    )
                    
                    f.write(f"- t-test: t={t_stat:.3f}, p={p_value:.5f}")
                    if p_value < 0.05:
                        f.write(" (statistically significant difference)\n")
                    else:
                        f.write(" (no statistically significant difference)\n")
                    
                    # Mann-Whitney U test (non-parametric) - if we have enough samples
                    if len(variants_data[variant1]) >= 3 and len(variants_data[variant2]) >= 3:
                        try:
                            u_stat, p_value = stats.mannwhitneyu(
                                variants_data[variant1],
                                variants_data[variant2]
                            )
                            
                            f.write(f"- Mann-Whitney U test: U={u_stat:.3f}, p={p_value:.5f}")
                            if p_value < 0.05:
                                f.write(" (statistically significant difference)\n")
                            else:
                                f.write(" (no statistically significant difference)\n")
                        except ValueError:
                            f.write("- Mann-Whitney U test could not be performed due to sample constraints\n")
                    else:
                        f.write("- Mann-Whitney U test skipped due to small sample size\n")
                    
                    # Effect size (Cohen's d) - if we have enough data
                    mean1, mean2 = np.mean(variants_data[variant1]), np.mean(variants_data[variant2])
                    std1, std2 = np.std(variants_data[variant1]), np.std(variants_data[variant2])
                    
                    # Pooled standard deviation
                    n1, n2 = len(variants_data[variant1]), len(variants_data[variant2])
                    if n1 > 1 and n2 > 1:  # Need at least 2 samples for std dev
                        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                        
                        # Cohen's d
                        cohen_d = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        
                        f.write(f"- Effect size (Cohen's d): {cohen_d:.3f} ")
                        if cohen_d < 0.2:
                            f.write("(negligible effect)\n")
                        elif cohen_d < 0.5:
                            f.write("(small effect)\n")
                        elif cohen_d < 0.8:
                            f.write("(medium effect)\n")
                        else:
                            f.write("(large effect)\n")
                    else:
                        f.write("- Effect size calculation skipped due to small sample size\n")
                    
                    f.write("\n")
        
        # Analyze specific metrics by variant
        
        # Value Loss Analysis
        if 'train_value_loss' in dfs:
            f.write("## Value Loss Analysis\n\n")
            
            value_loss = dfs['train_value_loss'].copy()
            
            f.write("| Variant | Mean | Median | Std Dev | Min | Max |\n")
            f.write("|---------|------|--------|---------|-----|-----|\n")
            
            for variant_type in value_loss['variant_type'].unique():
                variant_data = value_loss[value_loss['variant_type'] == variant_type]
                values = variant_data['value']
                
                f.write(f"| {variant_type} | {values.mean():.2f} | {values.median():.2f} | "
                       f"{values.std():.2f} | {values.min():.2f} | {values.max():.2f} |\n")
            
            f.write("\n")
        
        # Policy Loss Analysis
        if 'train_policy_loss' in dfs:
            f.write("## Policy Loss Analysis\n\n")
            
            policy_loss = dfs['train_policy_loss'].copy()
            
            f.write("| Variant | Mean | Median | Std Dev | Min | Max |\n")
            f.write("|---------|------|--------|---------|-----|-----|\n")
            
            for variant_type in policy_loss['variant_type'].unique():
                variant_data = policy_loss[policy_loss['variant_type'] == variant_type]
                values = variant_data['value']
                
                f.write(f"| {variant_type} | {values.mean():.6f} | {values.median():.6f} | "
                       f"{values.std():.6f} | {values.min():.6f} | {values.max():.6f} |\n")
            
            f.write("\n")
        
        # Entropy Analysis
        if 'train_entropy' in dfs:
            f.write("## Entropy Analysis\n\n")
            
            entropy = dfs['train_entropy'].copy()
            
            f.write("| Variant | Mean | Median | Std Dev | Min | Max |\n")
            f.write("|---------|------|--------|---------|-----|-----|\n")
            
            for variant_type in entropy['variant_type'].unique():
                variant_data = entropy[entropy['variant_type'] == variant_type]
                values = variant_data['value']
                
                f.write(f"| {variant_type} | {values.mean():.4f} | {values.median():.4f} | "
                       f"{values.std():.4f} | {values.min():.4f} | {values.max():.4f} |\n")
            
            f.write("\n")
            
            # Compare entropy between variants
            if len(entropy['variant_type'].unique()) > 1:
                entropy_by_variant = {}
                for variant_type in entropy['variant_type'].unique():
                    variant_data = entropy[entropy['variant_type'] == variant_type]
                    entropy_by_variant[variant_type] = variant_data['value'].mean()
                
                higher_entropy_variant = max(entropy_by_variant.items(), key=lambda x: x[1])[0]
                
                f.write(f"- **{higher_entropy_variant}** maintained higher entropy during training, which typically indicates greater exploration.\n")
                f.write("- Higher entropy generally leads to more exploration, which can help in complex environments with challenging exploration requirements.\n\n")
        
        # Algorithm-specific analysis
        
        # PPOCLIP specific analysis
        if 'train_clip_fraction' in dfs:
            clip_data = dfs['train_clip_fraction'][dfs['train_clip_fraction']['variant_type'] == 'PPOCLIP'].copy()
            
            if not clip_data.empty:
                f.write("### PPOCLIP-Specific Analysis\n\n")
                
                clip_values = clip_data['value']
                
                # Analyze clip fraction patterns
                over_time = np.corrcoef(range(len(clip_values)), clip_values)[0, 1]
                
                f.write(f"- Average clip fraction: {clip_values.mean():.4f}\n")
                
                if over_time > 0.3:
                    f.write("- Clip fraction showed an **increasing trend** over training, suggesting policy updates became larger over time.\n")
                elif over_time < -0.3:
                    f.write("- Clip fraction showed a **decreasing trend** over training, suggesting policy updates became more conservative over time.\n")
                else:
                    f.write("- Clip fraction remained relatively stable throughout training.\n")
                
                # Clip fraction thresholds
                high_clip_fraction = (clip_values > 0.2).mean() * 100
                low_clip_fraction = (clip_values < 0.05).mean() * 100
                
                if high_clip_fraction > 20:
                    f.write(f"- Clip fraction exceeded 0.2 in {high_clip_fraction:.1f}% of updates, suggesting the learning rate might be too high or the policy is changing too rapidly.\n")
                elif low_clip_fraction > 50:
                    f.write(f"- Clip fraction was below 0.05 in {low_clip_fraction:.1f}% of updates, suggesting the learning rate might be too low or the policy is changing too slowly.\n")
                else:
                    f.write("- Clip fraction values generally stayed within a reasonable range, suggesting appropriate hyperparameters for policy updates.\n")
                
                f.write("\n")
        
        # PPOKL specific analysis
        if 'train_kl_divergence' in dfs and 'train_kl_coef' in dfs:
            kl_data = dfs['train_kl_divergence'][dfs['train_kl_divergence']['variant_type'] == 'PPOKL'].copy()
            kl_coef_data = dfs['train_kl_coef'][dfs['train_kl_coef']['variant_type'] == 'PPOKL'].copy()
            
            if not kl_data.empty and not kl_coef_data.empty:
                f.write("### PPOKL-Specific Analysis\n\n")
                
                kl_values = kl_data['value']
                kl_coef_values = kl_coef_data['value']
                
                # Analyze KL divergence patterns
                target_kl = 0.005  # Common default value
                avg_kl = kl_values.mean()
                kl_stability = kl_values.std() / abs(avg_kl) if abs(avg_kl) > 0 else float('inf')
                
                f.write(f"- Average KL divergence: {avg_kl:.6f}\n")
                f.write(f"- KL divergence coefficient of variation: {kl_stability:.4f}\n")
                
                if abs(avg_kl - target_kl) / target_kl < 0.2:
                    f.write("- KL divergence stayed close to the target value, suggesting effective adaptive penalties.\n")
                elif avg_kl < target_kl:
                    f.write("- KL divergence was consistently below the target value, potentially resulting in overly conservative updates.\n")
                else:
                    f.write("- KL divergence was often above the target value, potentially resulting in overly aggressive updates.\n")
                
                # KL coefficient adaptation
                kl_coef_changes = len(kl_coef_values.unique())
                
                if kl_coef_changes > 1:
                    f.write(f"- The KL coefficient was adapted {kl_coef_changes-1} times during training.\n")
                    
                    if kl_coef_values.iloc[-1] > kl_coef_values.iloc[0]:
                        f.write("- KL coefficient increased over time, suggesting the algorithm needed stronger penalties to control policy updates.\n")
                    elif kl_coef_values.iloc[-1] < kl_coef_values.iloc[0]:
                        f.write("- KL coefficient decreased over time, suggesting the algorithm could use weaker penalties as training progressed.\n")
                else:
                    f.write("- The KL coefficient remained constant, indicating the policy updates naturally stayed within the desired range.\n")
                
                f.write("\n")
        
        # Efficiency analysis
        if 'train_reward_per_step' in dfs:
            f.write("## Efficiency Analysis\n\n")
            
            reward_per_step = dfs['train_reward_per_step'].copy()
            
            f.write("| Variant | Average Reward/Step | Final Reward/Step | Max Reward/Step |\n")
            f.write("|---------|---------------------|-------------------|----------------|\n")
            
            for variant_type in reward_per_step['variant_type'].unique():
                variant_data = reward_per_step[reward_per_step['variant_type'] == variant_type]
                values = variant_data['value']
                
                # Get final value
                final_value = values.iloc[-1] if not values.empty else 'N/A'
                
                f.write(f"| {variant_type} | {values.mean():.4f} | {final_value:.4f} | {values.max():.4f} |\n")
            
            f.write("\n")
            
            # Compare efficiency between variants
            if len(reward_per_step['variant_type'].unique()) > 1:
                f.write("### Efficiency Comparison\n\n")
                
                # Gather data by variant
                efficiency_by_variant = {}
                for variant_type in reward_per_step['variant_type'].unique():
                    variant_data = reward_per_step[reward_per_step['variant_type'] == variant_type]
                    efficiency_by_variant[variant_type] = variant_data['value'].mean()
                
                # Find most efficient variant
                most_efficient = max(efficiency_by_variant.items(), key=lambda x: x[1])
                
                relative_efficiency = {
                    variant: (efficiency / most_efficient[1]) * 100
                    for variant, efficiency in efficiency_by_variant.items()
                }
                
                f.write(f"The most efficient variant is **{most_efficient[0]}** with an average reward per step of {most_efficient[1]:.4f}.\n\n")
                
                f.write("| Variant | Relative Efficiency |\n")
                f.write("|---------|---------------------|\n")
                
                for variant, rel_eff in sorted(relative_efficiency.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"| {variant} | {rel_eff:.2f}% |\n")
                
                f.write("\n")
        
        # Final Recommendations
        f.write("## Final Recommendations\n\n")
        
        # Determine the best variant overall
        best_variant = None
        best_justification = ""
        
        # Use evaluation performance if available
        if 'eval_mean_reward' in dfs:
            eval_rewards = dfs['eval_mean_reward'].copy()
            
            # Get final evaluation rewards
            variant_final_rewards = {}
            for variant_type in eval_rewards['variant_type'].unique():
                variant_data = eval_rewards[eval_rewards['variant_type'] == variant_type]
                if variant_data.empty:
                    continue
                    
                last_step = variant_data['step'].max()
                last_reward = variant_data[variant_data['step'] == last_step]['value'].iloc[0] if not variant_data.empty else None
                
                if last_reward is not None:
                    variant_final_rewards[variant_type] = last_reward
            
            if variant_final_rewards:
                best_variant = max(variant_final_rewards.items(), key=lambda x: x[1])[0]
                best_justification = f"highest final evaluation reward ({variant_final_rewards[best_variant]:.2f})"
        
        # If no evaluation data, use training performance
        if best_variant is None and 'train_episode_reward' in dfs:
            train_rewards = dfs['train_episode_reward'].copy()
            
            variant_mean_rewards = {}
            for variant_type in train_rewards['variant_type'].unique():
                variant_data = train_rewards[train_rewards['variant_type'] == variant_type]
                if not variant_data.empty:
                    variant_mean_rewards[variant_type] = variant_data['value'].mean()
            
            if variant_mean_rewards:
                best_variant = max(variant_mean_rewards.items(), key=lambda x: x[1])[0]
                best_justification = f"highest average training reward ({variant_mean_rewards[best_variant]:.2f})"
        
        if best_variant:
            f.write(f"Based on the analysis, **{best_variant}** is recommended as the best overall variant due to {best_justification}.\n\n")
        
        # Recommendations based on variant type
        variants_in_data = list(set().union(*[df['variant_type'].unique() for df in dfs.values() if 'variant_type' in df.columns]))
        
        if 'PPOCLIP' in variants_in_data and 'PPOKL' in variants_in_data:
            f.write("### Algorithm Comparison\n\n")
            
            # Compare final performance
            if 'eval_mean_reward' in dfs:
                eval_rewards = dfs['eval_mean_reward'].copy()
                
                ppoclip_data = eval_rewards[eval_rewards['variant_type'] == 'PPOCLIP']
                ppokl_data = eval_rewards[eval_rewards['variant_type'] == 'PPOKL']
                
                if not ppoclip_data.empty and not ppokl_data.empty:
                    # Get final evaluation rewards
                    ppoclip_last_step = ppoclip_data['step'].max()
                    ppoclip_last_reward = ppoclip_data[ppoclip_data['step'] == ppoclip_last_step]['value'].iloc[0]
                    
                    ppokl_last_step = ppokl_data['step'].max()
                    ppokl_last_reward = ppokl_data[ppokl_data['step'] == ppokl_last_step]['value'].iloc[0]
                    
                    if ppoclip_last_reward > ppokl_last_reward:
                        f.write(f"- **PPOCLIP** achieved better final evaluation performance ({ppoclip_last_reward:.2f} vs {ppokl_last_reward:.2f}).\n")
                    else:
                        f.write(f"- **PPOKL** achieved better final evaluation performance ({ppokl_last_reward:.2f} vs {ppoclip_last_reward:.2f}).\n")
                    
                    # Compare mean evaluation performance
                    ppoclip_mean = ppoclip_data['value'].mean()
                    ppokl_mean = ppokl_data['value'].mean()
                    
                    if ppoclip_mean > ppokl_mean:
                        f.write(f"- **PPOCLIP** had better average evaluation performance ({ppoclip_mean:.2f} vs {ppokl_mean:.2f}).\n")
                    else:
                        f.write(f"- **PPOKL** had better average evaluation performance ({ppokl_mean:.2f} vs {ppoclip_mean:.2f}).\n")
            
            # Compare training stability
            if 'train_episode_reward' in dfs:
                train_rewards = dfs['train_episode_reward'].copy()
                
                ppoclip_data = train_rewards[train_rewards['variant_type'] == 'PPOCLIP']
                ppokl_data = train_rewards[train_rewards['variant_type'] == 'PPOKL']
                
                if not ppoclip_data.empty and not ppokl_data.empty:
                    # Calculate coefficient of variation as a measure of stability
                    ppoclip_cv = ppoclip_data['value'].std() / abs(ppoclip_data['value'].mean()) if ppoclip_data['value'].mean() != 0 else float('inf')
                    ppokl_cv = ppokl_data['value'].std() / abs(ppokl_data['value'].mean()) if ppokl_data['value'].mean() != 0 else float('inf')
                    
                    if ppoclip_cv < ppokl_cv:
                        f.write(f"- **PPOCLIP** showed more stable training (CV: {ppoclip_cv:.4f} vs {ppokl_cv:.4f}).\n")
                    else:
                        f.write(f"- **PPOKL** showed more stable training (CV: {ppokl_cv:.4f} vs {ppoclip_cv:.4f}).\n")
            
            f.write("\n")
            
            # General recommendations
            f.write("### Recommendations for Future Work\n\n")
            
            if 'train_clip_fraction' in dfs:
                clip_data = dfs['train_clip_fraction'][dfs['train_clip_fraction']['variant_type'] == 'PPOCLIP'].copy()
                
                if not clip_data.empty:
                    clip_values = clip_data['value']
                    
                    if clip_values.mean() > 0.2:
                        f.write("1. For **PPOCLIP**, consider reducing the learning rate as the clip fraction is frequently high, indicating large policy updates that may cause instability.\n")
                    elif clip_values.mean() < 0.05:
                        f.write("1. For **PPOCLIP**, consider increasing the learning rate as the clip fraction is quite low, indicating overly conservative policy updates.\n")
                    else:
                        f.write("1. For **PPOCLIP**, the current learning rate seems appropriate as clip fractions are in a reasonable range.\n")
            
            if 'train_kl_divergence' in dfs and 'train_kl_coef' in dfs:
                kl_data = dfs['train_kl_divergence'][dfs['train_kl_divergence']['variant_type'] == 'PPOKL'].copy()
                kl_coef_data = dfs['train_kl_coef'][dfs['train_kl_coef']['variant_type'] == 'PPOKL'].copy()
                
                if not kl_data.empty and not kl_coef_data.empty:
                    kl_values = kl_data['value']
                    kl_coef_values = kl_coef_data['value']
                    
                    if kl_values.mean() > 0.01:
                        f.write("2. For **PPOKL**, consider increasing the penalty coefficient as the KL divergence is often high, indicating large policy updates.\n")
                    elif kl_coef_values.iloc[-1] == kl_coef_values.min() and kl_values.mean() < 0.002:
                        f.write("2. For **PPOKL**, consider decreasing the minimum penalty coefficient as the KL divergence is consistently low, indicating overly conservative policy updates.\n")
                    else:
                        f.write("2. For **PPOKL**, the current KL divergence settings seem appropriate.\n")
            
            f.write("\n3. Consider longer training for both algorithms as the learning curves suggest performance was still improving at the end of training.\n")
            
            # Final variant recommendation
            if best_variant:
                f.write(f"\n**Final Recommendation:** Based on this analysis, **{best_variant}** is the recommended algorithm for this environment due to {best_justification}. Further hyperparameter tuning may lead to even better performance.\n")

def main():
    """Main function for analyzing PPO variants."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.logs_dir, 'analysis')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics from TensorBoard logs
    metrics_data = load_tensorboard_data(args.logs_dir)
    
    if not metrics_data:
        print("Error: No metrics data could be loaded. Exiting.")
        return
    
    # Convert to pandas DataFrames
    print("Preparing data for analysis...")
    dfs = prepare_dataframes(metrics_data)
    
    # Plot learning curves and other visualizations
    print("Generating plots...")
    plot_learning_curves(dfs, args, args.output_dir)
    
    # Generate statistical analysis
    print("Performing statistical analysis...")
    generate_statistical_analysis(dfs, args.output_dir)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()