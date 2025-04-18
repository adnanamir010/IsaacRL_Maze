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

def parse_arguments():
    """Parse command line arguments for the analysis script."""
    parser = argparse.ArgumentParser(description='Analyze RL algorithm comparison results')
    
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing the comparison results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis results (default: results_dir/analysis)')
    parser.add_argument('--smoothing', type=int, default=5,
                        help='Window size for smoothing plots (default: 5)')
    parser.add_argument('--figsize', type=str, default='10,6',
                        help='Figure size as width,height (default: 10,6)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures (default: 300)')
    parser.add_argument('--font-size', type=int, default=12,
                        help='Base font size for plots (default: 12)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-whitegrid',
                        help='Matplotlib style (default: seaborn-v0_8-whitegrid)')
    parser.add_argument('--no-confidence', action='store_true',
                        help='Disable confidence intervals in plots')
    parser.add_argument('--normalize-steps', action='store_true', default=True,
                        help='Normalize step counts for fair comparison (default: True)')
    
    return parser.parse_args()

def load_metrics(results_dir):
    """Load metrics from JSON files and config."""
    
    # Load PPO metrics
    ppo_path = os.path.join(results_dir, 'ppo_metrics.json')
    if os.path.exists(ppo_path):
        with open(ppo_path, 'r') as f:
            ppo_metrics = json.load(f)
    else:
        print(f"Warning: PPO metrics file not found at {ppo_path}")
        ppo_metrics = None
    
    # Load SAC metrics
    sac_path = os.path.join(results_dir, 'sac_metrics.json')
    if os.path.exists(sac_path):
        with open(sac_path, 'r') as f:
            sac_metrics = json.load(f)
    else:
        print(f"Warning: SAC metrics file not found at {sac_path}")
        sac_metrics = None
    
    # Load CSV if it exists
    csv_path = os.path.join(results_dir, 'comparison_metrics.csv')
    if os.path.exists(csv_path):
        comparison_df = pd.read_csv(csv_path)
    else:
        comparison_df = None
        print(f"Warning: Comparison CSV file not found at {csv_path}")
    
    # Load config
    config_path = os.path.join(results_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = None
        print(f"Warning: Config file not found at {config_path}")
    
    return ppo_metrics, sac_metrics, comparison_df, config

def prepare_data(ppo_metrics, sac_metrics, config):
    """Prepare data for analysis, handling different step counts."""
    
    metrics_data = {}
    
    # Determine the step counts from config if available
    ppo_steps = config.get('ppo_num_steps', config.get('num_steps', 0)) if config else 0
    sac_steps = config.get('num_steps', 0) if config else 0
    
    print(f"Step counts from config - PPO: {ppo_steps:,}, SAC: {sac_steps:,}")
    
    # Create DataFrames for each metric type
    if ppo_metrics:
        # Extract PPO algorithm type
        ppo_algorithm = ppo_metrics.get('algorithm', 'PPOCLIP')
        
        # Evaluation rewards
        ppo_eval_rewards = pd.DataFrame(ppo_metrics['eval_mean_rewards'], 
                                       columns=['steps', 'reward'])
        ppo_eval_rewards['algorithm'] = f"{ppo_algorithm} ({ppo_steps:,} steps)"
        ppo_eval_rewards['std'] = [std for _, std in ppo_metrics['eval_std_rewards']]
        
        # Normalize step counts if needed
        ppo_eval_rewards['normalized_steps'] = ppo_eval_rewards['steps'] / ppo_steps
        
        # Training rewards
        ppo_train_rewards = pd.DataFrame(ppo_metrics['train_rewards'],
                                        columns=['steps', 'reward'])
        ppo_train_rewards['algorithm'] = f"{ppo_algorithm} ({ppo_steps:,} steps)"
        ppo_train_rewards['normalized_steps'] = ppo_train_rewards['steps'] / ppo_steps
        
        # Losses
        ppo_value_losses = pd.DataFrame(ppo_metrics['value_losses'],
                                       columns=['steps', 'loss'])
        ppo_value_losses['type'] = 'value_loss'
        ppo_value_losses['algorithm'] = f"{ppo_algorithm} ({ppo_steps:,} steps)"
        ppo_value_losses['normalized_steps'] = ppo_value_losses['steps'] / ppo_steps
        
        ppo_policy_losses = pd.DataFrame(ppo_metrics['policy_losses'],
                                        columns=['steps', 'loss'])
        ppo_policy_losses['type'] = 'policy_loss'
        ppo_policy_losses['algorithm'] = f"{ppo_algorithm} ({ppo_steps:,} steps)"
        ppo_policy_losses['normalized_steps'] = ppo_policy_losses['steps'] / ppo_steps
        
        ppo_entropy_losses = pd.DataFrame(ppo_metrics['entropy_losses'],
                                         columns=['steps', 'loss'])
        ppo_entropy_losses['type'] = 'entropy_loss'
        ppo_entropy_losses['algorithm'] = f"{ppo_algorithm} ({ppo_steps:,} steps)"
        ppo_entropy_losses['normalized_steps'] = ppo_entropy_losses['steps'] / ppo_steps
        
        # Combine losses
        ppo_losses = pd.concat([ppo_value_losses, ppo_policy_losses, ppo_entropy_losses])
        
        # Add algorithm-specific metrics
        if 'clip_fractions' in ppo_metrics:
            ppo_clip_fractions = pd.DataFrame(ppo_metrics['clip_fractions'],
                                            columns=['steps', 'fraction'])
            ppo_clip_fractions['algorithm'] = f"{ppo_algorithm} ({ppo_steps:,} steps)"
            ppo_clip_fractions['normalized_steps'] = ppo_clip_fractions['steps'] / ppo_steps
            metrics_data['ppo_clip_fractions'] = ppo_clip_fractions
        
        if 'kl_divergences' in ppo_metrics:
            ppo_kl_divergences = pd.DataFrame(ppo_metrics['kl_divergences'],
                                            columns=['steps', 'kl'])
            ppo_kl_divergences['algorithm'] = f"{ppo_algorithm} ({ppo_steps:,} steps)"
            ppo_kl_divergences['normalized_steps'] = ppo_kl_divergences['steps'] / ppo_steps
            metrics_data['ppo_kl_divergences'] = ppo_kl_divergences
        
        metrics_data['ppo_eval_rewards'] = ppo_eval_rewards
        metrics_data['ppo_train_rewards'] = ppo_train_rewards
        metrics_data['ppo_losses'] = ppo_losses
    
    if sac_metrics:
        # Determine SAC variants
        critic_type = sac_metrics.get('critic_type', 'Twin')
        entropy_type = sac_metrics.get('entropy_type', 'Adaptive')
        
        # Evaluation rewards
        sac_eval_rewards = pd.DataFrame(sac_metrics['eval_mean_rewards'], 
                                       columns=['steps', 'reward'])
        sac_eval_rewards['algorithm'] = f"SAC {critic_type}Critic {entropy_type} ({sac_steps:,} steps)"
        sac_eval_rewards['std'] = [std for _, std in sac_metrics['eval_std_rewards']]
        sac_eval_rewards['normalized_steps'] = sac_eval_rewards['steps'] / sac_steps
        
        # Training rewards
        sac_train_rewards = pd.DataFrame(sac_metrics['train_rewards'],
                                        columns=['steps', 'reward'])
        sac_train_rewards['algorithm'] = f"SAC {critic_type}Critic {entropy_type} ({sac_steps:,} steps)"
        sac_train_rewards['normalized_steps'] = sac_train_rewards['steps'] / sac_steps
        
        # Losses
        sac_critic1_losses = pd.DataFrame(sac_metrics['critic_1_losses'],
                                         columns=['steps', 'loss'])
        sac_critic1_losses['type'] = 'critic_1_loss'
        sac_critic1_losses['algorithm'] = f"SAC {critic_type}Critic {entropy_type} ({sac_steps:,} steps)"
        sac_critic1_losses['normalized_steps'] = sac_critic1_losses['steps'] / sac_steps
        
        sac_critic2_losses = pd.DataFrame(sac_metrics['critic_2_losses'],
                                         columns=['steps', 'loss'])
        sac_critic2_losses['type'] = 'critic_2_loss'
        sac_critic2_losses['algorithm'] = f"SAC {critic_type}Critic {entropy_type} ({sac_steps:,} steps)"
        sac_critic2_losses['normalized_steps'] = sac_critic2_losses['steps'] / sac_steps
        
        sac_policy_losses = pd.DataFrame(sac_metrics['policy_losses'],
                                        columns=['steps', 'loss'])
        sac_policy_losses['type'] = 'policy_loss'
        sac_policy_losses['algorithm'] = f"SAC {critic_type}Critic {entropy_type} ({sac_steps:,} steps)"
        sac_policy_losses['normalized_steps'] = sac_policy_losses['steps'] / sac_steps
        
        sac_alpha_losses = pd.DataFrame(sac_metrics['alpha_losses'],
                                      columns=['steps', 'loss'])
        sac_alpha_losses['type'] = 'alpha_loss'
        sac_alpha_losses['algorithm'] = f"SAC {critic_type}Critic {entropy_type} ({sac_steps:,} steps)"
        sac_alpha_losses['normalized_steps'] = sac_alpha_losses['steps'] / sac_steps
        
        # Combine losses
        sac_losses = pd.concat([sac_critic1_losses, sac_critic2_losses, 
                              sac_policy_losses, sac_alpha_losses])
        
        # Alpha values
        sac_alphas = pd.DataFrame(sac_metrics['alphas'],
                                 columns=['steps', 'alpha'])
        sac_alphas['algorithm'] = f"SAC {critic_type}Critic {entropy_type} ({sac_steps:,} steps)"
        sac_alphas['normalized_steps'] = sac_alphas['steps'] / sac_steps
        
        metrics_data['sac_eval_rewards'] = sac_eval_rewards
        metrics_data['sac_train_rewards'] = sac_train_rewards
        metrics_data['sac_losses'] = sac_losses
        metrics_data['sac_alphas'] = sac_alphas
    
    # Combine evaluation rewards
    if ppo_metrics and sac_metrics:
        eval_rewards = pd.concat([metrics_data['ppo_eval_rewards'], 
                                metrics_data['sac_eval_rewards']])
        metrics_data['eval_rewards'] = eval_rewards
        
        # Combine training rewards
        train_rewards = pd.concat([metrics_data['ppo_train_rewards'], 
                                  metrics_data['sac_train_rewards']])
        metrics_data['train_rewards'] = train_rewards
    
    # Add the step counts to the metadata
    metrics_data['step_info'] = {
        'ppo_steps': ppo_steps,
        'sac_steps': sac_steps,
        'max_steps': max(ppo_steps, sac_steps)
    }
    
    return metrics_data

def apply_smoothing(data, window_size=5):
    """Apply moving average smoothing to data."""
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_learning_curves(metrics_data, args, output_dir):
    """Plot learning curves comparison, handling different step counts."""
    
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
    
    # Get step info for proper plotting
    step_info = metrics_data.get('step_info', {})
    ppo_steps = step_info.get('ppo_steps', 0)
    sac_steps = step_info.get('sac_steps', 0)
    
    # Flag to use normalized steps for fair comparison
    use_normalized = args.normalize_steps and ppo_steps != sac_steps and ppo_steps > 0 and sac_steps > 0
    step_column = 'normalized_steps' if use_normalized else 'steps'
    
    # 1. Plot evaluation rewards with confidence intervals
    if 'eval_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # Group by algorithm and steps
        eval_rewards = metrics_data['eval_rewards']
        
        # Plot each algorithm's learning curve
        for algo in eval_rewards['algorithm'].unique():
            algo_data = eval_rewards[eval_rewards['algorithm'] == algo].copy()
            
            # Sort by steps
            algo_data = algo_data.sort_values(step_column)
            
            # Scale the x-axis based on step counts if using absolute values
            if not use_normalized:
                x_values = algo_data[step_column].values
            else:
                x_values = algo_data[step_column].values * 100  # As percentage
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_rewards = apply_smoothing(algo_data['reward'], args.smoothing)
                smoothed_stds = apply_smoothing(algo_data['std'], args.smoothing)
            else:
                smoothed_rewards = algo_data['reward']
                smoothed_stds = algo_data['std']
            
            # Plot mean
            line = ax.plot(x_values, smoothed_rewards, 
                          label=algo, linewidth=2)
            
            # Add confidence interval
            if not args.no_confidence:
                ax.fill_between(
                    x_values,
                    smoothed_rewards - smoothed_stds,
                    smoothed_rewards + smoothed_stds,
                    alpha=0.2,
                    color=line[0].get_color()
                )
        
        # Set x-axis label based on normalization
        if use_normalized:
            ax.set_xlabel('Training Progress (%)')
            ax.set_xlim([0, 100])
        else:
            ax.set_xlabel('Environment Steps')
        
        ax.set_ylabel('Mean Evaluation Reward')
        if use_normalized:
            ax.set_title('Evaluation Rewards vs Training Progress')
        else:
            ax.set_title('Evaluation Rewards vs Steps')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for maximum performance
        for algo in eval_rewards['algorithm'].unique():
            algo_data = eval_rewards[eval_rewards['algorithm'] == algo]
            max_idx = algo_data['reward'].idxmax()
            if use_normalized:
                max_x = algo_data.loc[max_idx, step_column] * 100  # As percentage
            else:
                max_x = algo_data.loc[max_idx, step_column]
            max_reward = algo_data.loc[max_idx, 'reward']
            
            # Add annotation with an arrow
            ax.annotate(f'Max: {max_reward:.2f}',
                      xy=(max_x, max_reward),
                      xytext=(10, 10),
                      textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eval_rewards_comparison.png'))
        plt.close()
    
    # 2. Plot training rewards
    if 'train_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # For training rewards, we'll aggregate by step bins to make the plot cleaner
        train_rewards = metrics_data['train_rewards'].copy()
        
        # Determine bin size based on step normalization
        if use_normalized:
            # Use percentage bins
            bin_size = 0.01  # 1% increments
            train_rewards['step_bin'] = (train_rewards[step_column] // bin_size) * bin_size
        else:
            # Determine bin size based on total steps
            max_steps = max(train_rewards['steps'])
            bin_size = max(1, max_steps // 100)  # About 100 points on the x-axis
            train_rewards['step_bin'] = (train_rewards['steps'] // bin_size) * bin_size
        
        # Group by algorithm and step bin
        grouped = train_rewards.groupby(['algorithm', 'step_bin']).agg({
            'reward': ['mean', 'std']
        }).reset_index()
        
        # Flatten multi-index columns
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
        
        # Plot for each algorithm
        for algo in grouped['algorithm'].unique():
            algo_data = grouped[grouped['algorithm'] == algo]
            
            # Sort by step bin
            algo_data = algo_data.sort_values('step_bin')
            
            # Scale x-axis for normalized steps
            if use_normalized:
                x_values = algo_data['step_bin'] * 100  # As percentage
            else:
                x_values = algo_data['step_bin']
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_rewards = apply_smoothing(algo_data['reward_mean'], args.smoothing)
                if 'reward_std' in algo_data.columns:
                    smoothed_stds = apply_smoothing(algo_data['reward_std'], args.smoothing)
                else:
                    smoothed_stds = None
            else:
                smoothed_rewards = algo_data['reward_mean']
                smoothed_stds = algo_data['reward_std'] if 'reward_std' in algo_data.columns else None
            
            # Plot mean
            line = ax.plot(x_values, smoothed_rewards, 
                          label=algo, linewidth=2, alpha=0.8)
            
            # Add confidence interval
            if not args.no_confidence and smoothed_stds is not None:
                ax.fill_between(
                    x_values,
                    smoothed_rewards - smoothed_stds,
                    smoothed_rewards + smoothed_stds,
                    alpha=0.1,
                    color=line[0].get_color()
                )
        
        # Set x-axis label based on normalization
        if use_normalized:
            ax.set_xlabel('Training Progress (%)')
            ax.set_xlim([0, 100])
        else:
            ax.set_xlabel('Environment Steps')
        
        ax.set_ylabel('Mean Training Episode Reward')
        if use_normalized:
            ax.set_title('Training Rewards vs Progress')
        else:
            ax.set_title('Training Rewards vs Steps')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_rewards_comparison.png'))
        plt.close()
    
    # 3. Plot loss curves for each algorithm
    # 3a. PPO Losses
    if 'ppo_losses' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        ppo_losses = metrics_data['ppo_losses'].copy()
        
        # Bin losses by steps
        if use_normalized:
            bin_size = 0.01  # 1% increments
            ppo_losses['step_bin'] = (ppo_losses[step_column] // bin_size) * bin_size
            x_label = 'Training Progress (%)'
        else:
            max_steps = max(ppo_losses['steps'])
            bin_size = max(1, max_steps // 100)
            ppo_losses['step_bin'] = (ppo_losses['steps'] // bin_size) * bin_size
            x_label = 'Environment Steps'
        
        # Group by loss type and step bin
        grouped = ppo_losses.groupby(['type', 'step_bin']).agg({
            'loss': 'mean'
        }).reset_index()
        
        # Plot each loss type
        for loss_type in grouped['type'].unique():
            type_data = grouped[grouped['type'] == loss_type]
            
            # Sort by step bin
            type_data = type_data.sort_values('step_bin')
            
            # Scale x-axis for normalized steps
            if use_normalized:
                x_values = type_data['step_bin'] * 100  # As percentage
            else:
                x_values = type_data['step_bin']
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_losses = apply_smoothing(type_data['loss'], args.smoothing)
            else:
                smoothed_losses = type_data['loss']
            
            # Normalize losses for better comparison
            if loss_type == 'entropy_loss':
                # Usually entropy loss is much smaller, scale it up
                scale_factor = max(1, abs(grouped[grouped['type'] != 'entropy_loss']['loss'].median() /
                                      type_data['loss'].median()))
                scaled_losses = smoothed_losses * scale_factor
                ax.plot(x_values, scaled_losses, 
                       label=f'{loss_type} (scaled)', linewidth=2)
            else:
                ax.plot(x_values, smoothed_losses, 
                       label=loss_type, linewidth=2)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Loss')
        ax.set_title('PPO Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit for normalized steps
        if use_normalized:
            ax.set_xlim([0, 100])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ppo_losses.png'))
        plt.close()
    
    # 3b. SAC Losses
    if 'sac_losses' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        sac_losses = metrics_data['sac_losses'].copy()
        
        # Bin losses by steps
        if use_normalized:
            bin_size = 0.01  # 1% increments
            sac_losses['step_bin'] = (sac_losses[step_column] // bin_size) * bin_size
            x_label = 'Training Progress (%)'
        else:
            max_steps = max(sac_losses['steps'])
            bin_size = max(1, max_steps // 100)
            sac_losses['step_bin'] = (sac_losses['steps'] // bin_size) * bin_size
            x_label = 'Environment Steps'
        
        # Group by loss type and step bin
        grouped = sac_losses.groupby(['type', 'step_bin']).agg({
            'loss': 'mean'
        }).reset_index()
        
        # Plot each loss type
        for loss_type in grouped['type'].unique():
            type_data = grouped[grouped['type'] == loss_type]
            
            # Sort by step bin
            type_data = type_data.sort_values('step_bin')
            
            # Scale x-axis for normalized steps
            if use_normalized:
                x_values = type_data['step_bin'] * 100  # As percentage
            else:
                x_values = type_data['step_bin']
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_losses = apply_smoothing(type_data['loss'], args.smoothing)
            else:
                smoothed_losses = type_data['loss']
            
            # Plot
            ax.plot(x_values, smoothed_losses, 
                   label=loss_type, linewidth=2)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Loss')
        ax.set_title('SAC Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit for normalized steps
        if use_normalized:
            ax.set_xlim([0, 100])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sac_losses.png'))
        plt.close()
    
    # 4. Plot SAC alpha parameter
    if 'sac_alphas' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        sac_alphas = metrics_data['sac_alphas'].copy()
        
        # Sort by steps
        sac_alphas = sac_alphas.sort_values(step_column)
        
        # Scale x-axis for normalized steps
        if use_normalized:
            x_values = sac_alphas[step_column] * 100  # As percentage
            x_label = 'Training Progress (%)'
        else:
            x_values = sac_alphas['steps']
            x_label = 'Environment Steps'
        
        # Apply smoothing
        if args.smoothing > 1:
            smoothed_alphas = apply_smoothing(sac_alphas['alpha'], args.smoothing)
        else:
            smoothed_alphas = sac_alphas['alpha']
        
        # Plot
        ax.plot(x_values, smoothed_alphas, 
               label='alpha', linewidth=2)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Alpha Value')
        ax.set_title('SAC Temperature Parameter (Alpha)')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit for normalized steps
        if use_normalized:
            ax.set_xlim([0, 100])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sac_alpha.png'))
        plt.close()
    
    # 5. Performance comparison bar chart
    if 'eval_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # Get final evaluation rewards
        final_rewards = []
        
        for algo in metrics_data['eval_rewards']['algorithm'].unique():
            algo_data = metrics_data['eval_rewards'][metrics_data['eval_rewards']['algorithm'] == algo]
            
            # Get last recorded reward
            last_idx = algo_data['steps'].idxmax()
            final_reward = algo_data.loc[last_idx, 'reward']
            final_std = algo_data.loc[last_idx, 'std']
            
            final_rewards.append({
                'algorithm': algo,
                'reward': final_reward,
                'std': final_std
            })
        
        # Convert to DataFrame
        final_df = pd.DataFrame(final_rewards)
        
        # Sort by reward
        final_df = final_df.sort_values('reward', ascending=False)
        
        # Create bar plot
        bars = ax.bar(
            final_df['algorithm'],
            final_df['reward'],
            yerr=final_df['std'],
            capsize=10,
            color=['blue', 'red'],
            alpha=0.7
        )
        
        # Add value labels
        for bar, reward in zip(bars, final_df['reward']):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{reward:.2f}",
                ha='center',
                va='bottom'
            )
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Final Mean Reward')
        ax.set_title('Final Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust figure for long algorithm names
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'final_performance.png'))
        plt.close()
    
    # 6. Learning progress visualization (normalized to percentage)
    if 'eval_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        eval_rewards = metrics_data['eval_rewards'].copy()
        
        # Get maximum reward value overall
        max_reward = eval_rewards['reward'].max()
        
        # For each algorithm, calculate percentage of max reward achieved over time
        for algo in eval_rewards['algorithm'].unique():
            algo_data = eval_rewards[eval_rewards['algorithm'] == algo]
            
            # Sort by steps
            algo_data = algo_data.sort_values(step_column)
            
            # Scale x-axis for normalized steps
            if use_normalized:
                x_values = algo_data[step_column] * 100  # As percentage
            else:
                x_values = algo_data[step_column]
            
            # Calculate percentage of max reward
            algo_data['progress'] = algo_data['reward'] / max_reward * 100
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_progress = apply_smoothing(algo_data['progress'], args.smoothing)
            else:
                smoothed_progress = algo_data['progress']
            
            # Plot
            ax.plot(x_values, smoothed_progress, 
                   label=algo, linewidth=2)
        
        # Set x-axis label based on normalization
        if use_normalized:
            ax.set_xlabel('Training Progress (%)')
            ax.set_xlim([0, 100])
        else:
            ax.set_xlabel('Environment Steps')
        
        ax.set_ylabel('Percentage of Maximum Reward')
        ax.set_title('Learning Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add 25%, 50%, 75% horizontal lines
        for pct in [25, 50, 75]:
            ax.axhline(y=pct, color='gray', linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[1] * 0.02, pct + 2, f"{pct}%", 
                   color='gray', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_progress.png'))
        plt.close()
    
    # 7. Normalized sample efficiency comparison
    if 'eval_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        eval_rewards = metrics_data['eval_rewards'].copy()
        
        # For each algorithm, find the number of steps needed to reach certain performance thresholds
        thresholds = [0.25, 0.5, 0.75, 0.9]
        
        # First, find the maximum reward achieved by any algorithm
        max_reward = eval_rewards['reward'].max()
        threshold_values = [max_reward * t for t in thresholds]
        
        # Dictionary to store steps needed to reach each threshold
        steps_to_threshold = {algo: [] for algo in eval_rewards['algorithm'].unique()}
        
        for algo in eval_rewards['algorithm'].unique():
            algo_data = eval_rewards[eval_rewards['algorithm'] == algo].sort_values('steps')
            
            for threshold, threshold_value in zip(thresholds, threshold_values):
                # Find first time the algorithm exceeds this threshold
                above_threshold = algo_data[algo_data['reward'] >= threshold_value]
                
                if not above_threshold.empty:
                    # Get the steps at which it first exceeded the threshold
                    first_crossing = above_threshold.iloc[0]
                    
                    # Use normalized or absolute step count
                    if use_normalized:
                        steps_to_threshold[algo].append((threshold, first_crossing[step_column] * 100))  # As percentage
                    else:
                        steps_to_threshold[algo].append((threshold, first_crossing['steps']))
                else:
                    # Algorithm never reached this threshold
                    steps_to_threshold[algo].append((threshold, np.nan))
        
        # Create bar plot for each threshold
        width = 0.35  # width of the bars
        x = np.arange(len(thresholds))
        
        colors = ['blue', 'red']
        
        for i, algo in enumerate(steps_to_threshold.keys()):
            # Extract steps for each threshold
            values = [steps for _, steps in steps_to_threshold[algo]]
            
            # Plot bars
            bars = ax.bar(x + i*width - width/2, values, width, 
                         label=algo, color=colors[i], alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    height = bar.get_height()
                    if use_normalized:
                        # Show as percentage
                        ax.text(bar.get_x() + bar.get_width()/2, height,
                               f"{value:.1f}%",
                               ha='center', va='bottom',
                               rotation=45, fontsize=args.font_size-2)
                    else:
                        # Format as number with commas
                        ax.text(bar.get_x() + bar.get_width()/2, height,
                               f"{int(value):,}",
                               ha='center', va='bottom',
                               rotation=45, fontsize=args.font_size-2)
        
        # Set x-axis labels and ticks
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds])
        
        if use_normalized:
            ax.set_ylabel('Training Progress to Reach Threshold (%)')
            ax.set_title('Sample Efficiency Comparison (Normalized)')
        else:
            ax.set_ylabel('Steps to Reach Threshold')
            ax.set_title('Sample Efficiency Comparison')
            
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_efficiency.png'))
        plt.close()
    
    # 8. PPO-specific plots
    # 8a. Clip fractions (for PPOCLIP)
    if 'ppo_clip_fractions' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        clip_data = metrics_data['ppo_clip_fractions'].copy()
        clip_data = clip_data.sort_values(step_column)
        
        # Scale x-axis for normalized steps
        if use_normalized:
            x_values = clip_data[step_column] * 100  # As percentage
            x_label = 'Training Progress (%)'
        else:
            x_values = clip_data['steps']
            x_label = 'Environment Steps'
        
        # Apply smoothing
        if args.smoothing > 1:
            smoothed_clip = apply_smoothing(clip_data['fraction'], args.smoothing)
        else:
            smoothed_clip = clip_data['fraction']
        
        # Plot
        ax.plot(x_values, smoothed_clip, 'b-', linewidth=2, label='Clip Fraction')
        
        # Add reference lines
        ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10% Threshold')
        ax.axhline(y=0.2, color='orange', linestyle='-.', alpha=0.5, label='20% Threshold')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Clip Fraction')
        ax.set_title('PPO Clip Fraction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit for normalized steps
        if use_normalized:
            ax.set_xlim([0, 100])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ppo_clip_fraction.png'))
        plt.close()
    
    # 8b. KL divergences (for PPOKL)
    if 'ppo_kl_divergences' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        kl_data = metrics_data['ppo_kl_divergences'].copy()
        kl_data = kl_data.sort_values(step_column)
        
        # Scale x-axis for normalized steps
        if use_normalized:
            x_values = kl_data[step_column] * 100  # As percentage
            x_label = 'Training Progress (%)'
        else:
            x_values = kl_data['steps']
            x_label = 'Environment Steps'
        
        # Apply smoothing
        if args.smoothing > 1:
            smoothed_kl = apply_smoothing(kl_data['kl'], args.smoothing)
        else:
            smoothed_kl = kl_data['kl']
        
        # Plot
        ax.plot(x_values, smoothed_kl, 'b-', linewidth=2, label='KL Divergence')
        
        # Add target KL reference line if in config
        if 'config' in locals() and config and 'kl_target' in config:
            kl_target = config['kl_target']
            ax.axhline(y=kl_target, color='r', linestyle='--', alpha=0.7, 
                      label=f'Target KL: {kl_target}')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('KL Divergence')
        ax.set_title('PPO KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limit for normalized steps
        if use_normalized:
            ax.set_xlim([0, 100])
        
        # Use log scale for KL values
        if np.max(smoothed_kl) / np.min(smoothed_kl) > 100:
            ax.set_yscale('log')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ppo_kl_divergence.png'))
        plt.close()
    
    # 9. Create comparison dashboard
    # This is a summary figure showing the most important metrics side by side
    fig = plt.figure(figsize=(15, 10), dpi=args.dpi)
    
    # 9a. Evaluation rewards
    if 'eval_rewards' in metrics_data:
        ax1 = fig.add_subplot(221)
        
        for algo in metrics_data['eval_rewards']['algorithm'].unique():
            algo_data = metrics_data['eval_rewards'][metrics_data['eval_rewards']['algorithm'] == algo].copy()
            algo_data = algo_data.sort_values(step_column)
            
            # Scale x-axis for normalized steps
            if use_normalized:
                x_values = algo_data[step_column] * 100  # As percentage
            else:
                x_values = algo_data[step_column]
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_rewards = apply_smoothing(algo_data['reward'], args.smoothing)
            else:
                smoothed_rewards = algo_data['reward']
            
            ax1.plot(x_values, smoothed_rewards, 
                    label=algo, linewidth=2)
        
        if use_normalized:
            ax1.set_xlabel('Training Progress (%)')
            ax1.set_xlim([0, 100])
        else:
            ax1.set_xlabel('Environment Steps')
        
        ax1.set_ylabel('Mean Evaluation Reward')
        ax1.set_title('Evaluation Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 9b. Sample efficiency
    if 'eval_rewards' in metrics_data:
        ax2 = fig.add_subplot(222)
        
        # Use the sample efficiency calculations from before
        width = 0.35
        x = np.arange(len(thresholds))
        
        for i, algo in enumerate(steps_to_threshold.keys()):
            values = [steps for _, steps in steps_to_threshold[algo]]
            ax2.bar(x + i*width - width/2, values, width, 
                   label=algo, color=colors[i], alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{int(t*100)}%" for t in thresholds])
        
        if use_normalized:
            ax2.set_ylabel('Training Progress (%)')
        else:
            ax2.set_ylabel('Steps to Reach Threshold')
            
        ax2.set_title('Sample Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 9c. Final performance
    if 'eval_rewards' in metrics_data:
        ax3 = fig.add_subplot(223)
        
        # Use the final rewards calculated earlier
        ax3.bar(
            final_df['algorithm'],
            final_df['reward'],
            yerr=final_df['std'],
            capsize=10,
            color=['blue', 'red'],
            alpha=0.7
        )
        
        ax3.set_ylabel('Final Mean Reward')
        ax3.set_title('Final Performance')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.get_xticklabels(), rotation=15, ha='right')
    
    # 9d. Learning progress (normalized)
    if 'eval_rewards' in metrics_data:
        ax4 = fig.add_subplot(224)
        
        for algo in eval_rewards['algorithm'].unique():
            algo_data = eval_rewards[eval_rewards['algorithm'] == algo].copy()
            algo_data = algo_data.sort_values(step_column)
            
            # Scale x-axis for normalized steps
            if use_normalized:
                x_values = algo_data[step_column] * 100  # As percentage
            else:
                x_values = algo_data[step_column]
            
            # Calculate percentage of max reward
            algo_data['progress'] = algo_data['reward'] / max_reward * 100
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_progress = apply_smoothing(algo_data['progress'], args.smoothing)
            else:
                smoothed_progress = algo_data['progress']
            
            ax4.plot(x_values, smoothed_progress, 
                    label=algo, linewidth=2)
        
        if use_normalized:
            ax4.set_xlabel('Training Progress (%)')
            ax4.set_xlim([0, 100])
        else:
            ax4.set_xlabel('Environment Steps')
            
        ax4.set_ylabel('% of Maximum Reward')
        ax4.set_title('Learning Progress')
        ax4.grid(True, alpha=0.3)
        
        # Add 50% and 75% reference lines
        for pct in [50, 75]:
            ax4.axhline(y=pct, color='gray', linestyle='--', alpha=0.5)
            ax4.text(ax4.get_xlim()[1] * 0.02, pct + 2, f"{pct}%", 
                    color='gray', alpha=0.7)
    
    # Main title explaining the comparison
    if use_normalized:
        fig.suptitle(f'PPO vs SAC Comparison (Normalized by Training Progress)', fontsize=16)
    else:
        fig.suptitle(f'PPO ({ppo_steps:,} steps) vs SAC ({sac_steps:,} steps) Comparison', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'algorithm_comparison_dashboard.png'))
    plt.close()

def generate_statistical_analysis(metrics_data, config, output_dir):
    """Generate statistical analysis of the results."""
    
    # Create a text file to store the analysis
    stats_file = os.path.join(output_dir, 'statistical_analysis.txt')
    
    # Get step info
    step_info = metrics_data.get('step_info', {})
    ppo_steps = step_info.get('ppo_steps', 0)
    sac_steps = step_info.get('sac_steps', 0)
    
    with open(stats_file, 'w') as f:
        f.write("# Statistical Analysis of Algorithm Performance\n\n")
        
        # Write step information
        f.write("## Training Configuration\n\n")
        if ppo_steps > 0:
            f.write(f"- PPO trained for {ppo_steps:,} steps\n")
        if sac_steps > 0:
            f.write(f"- SAC trained for {sac_steps:,} steps\n")
            
        # Note about different step counts
        if ppo_steps != sac_steps and ppo_steps > 0 and sac_steps > 0:
            f.write(f"\n**Note:** PPO trained for {ppo_steps/sac_steps:.1f}x more steps than SAC. ")
            f.write("This analysis considers this difference.\n\n")
        
        if 'eval_rewards' in metrics_data:
            f.write("## Evaluation Rewards Analysis\n\n")
            
            # Group evaluation rewards by algorithm
            algorithms = metrics_data['eval_rewards']['algorithm'].unique()
            
            rewards_by_algo = {}
            for algo in algorithms:
                algo_data = metrics_data['eval_rewards'][metrics_data['eval_rewards']['algorithm'] == algo]
                rewards_by_algo[algo] = algo_data['reward'].values
                
                # Also calculate progress-normalized metrics if step counts differ
                if ppo_steps != sac_steps and ppo_steps > 0 and sac_steps > 0:
                    # Get normalized data
                    algo_data_norm = algo_data.copy()
                    normalized_steps = algo_data_norm['normalized_steps'] * 100  # As percentage
                    
                    # Store for later analysis
                    rewards_by_algo[f"{algo}_normalized"] = algo_data_norm['reward'].values
                    rewards_by_algo[f"{algo}_progress"] = normalized_steps
            
            # Basic statistics for each algorithm
            f.write("### Basic Statistics\n\n")
            f.write("| Algorithm | Mean | Median | Std Dev | Min | Max | Final |\n")
            f.write("|-----------|------|--------|---------|-----|-----|-------|\n")
            
            for algo, rewards in rewards_by_algo.items():
                if "_normalized" in algo or "_progress" in algo:
                    continue  # Skip normalized data in this table
                
                last_reward = rewards[-1]
                f.write(f"| {algo} | {np.mean(rewards):.2f} | {np.median(rewards):.2f} | "
                       f"{np.std(rewards):.2f} | {np.min(rewards):.2f} | {np.max(rewards):.2f} | "
                       f"{last_reward:.2f} |\n")
            
            f.write("\n")
            
            # Comparative statistics if we have multiple algorithms
            if len(algorithms) > 1:
                f.write("### Comparative Analysis\n\n")
                
                # Progress-normalized metrics if step counts differ
                if ppo_steps != sac_steps and ppo_steps > 0 and sac_steps > 0:
                    f.write("#### Training Progress Comparison\n\n")
                    f.write("Since PPO and SAC were trained for different numbers of steps, it's important to compare them at equivalent points in their training progress.\n\n")
                    
                    # Compare at 25%, 50%, 75%, and 100% of training
                    progress_points = [25, 50, 75, 100]
                    
                    f.write("| Progress | ")
                    for algo in algorithms:
                        f.write(f"{algo} | ")
                    f.write("\n")
                    
                    f.write("|----------|")
                    for _ in algorithms:
                        f.write("-----------|")
                    f.write("\n")
                    
                    for progress in progress_points:
                        f.write(f"| {progress}% | ")
                        
                        for algo in algorithms:
                            # Find the reward at this progress point
                            algo_progress = rewards_by_algo[f"{algo}_progress"]
                            algo_rewards = rewards_by_algo[f"{algo}_normalized"]
                            
                            # Find closest progress point
                            closest_idx = np.argmin(np.abs(algo_progress - progress))
                            reward_at_progress = algo_rewards[closest_idx]
                            
                            f.write(f"{reward_at_progress:.2f} | ")
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                # Perform statistical tests
                f.write("#### Statistical Significance Tests\n\n")
                
                # Organize algorithms for pairwise comparisons
                algo_pairs = []
                for i in range(len(algorithms)):
                    for j in range(i+1, len(algorithms)):
                        algo_pairs.append((algorithms[i], algorithms[j]))
                
                for algo1, algo2 in algo_pairs:
                    f.write(f"**{algo1} vs {algo2}**\n\n")
                    
                    # T-test for comparing means
                    t_stat, p_value = stats.ttest_ind(
                        rewards_by_algo[algo1], 
                        rewards_by_algo[algo2],
                        equal_var=False  # Welch's t-test (doesn't assume equal variance)
                    )
                    
                    f.write(f"- t-test: t={t_stat:.3f}, p={p_value:.5f}")
                    if p_value < 0.05:
                        f.write(" (statistically significant difference)\n")
                    else:
                        f.write(" (no statistically significant difference)\n")
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, p_value = stats.mannwhitneyu(
                        rewards_by_algo[algo1], 
                        rewards_by_algo[algo2]
                    )
                    
                    f.write(f"- Mann-Whitney U test: U={u_stat:.3f}, p={p_value:.5f}")
                    if p_value < 0.05:
                        f.write(" (statistically significant difference)\n")
                    else:
                        f.write(" (no statistically significant difference)\n")
                    
                    # Effect size (Cohen's d)
                    mean1, mean2 = np.mean(rewards_by_algo[algo1]), np.mean(rewards_by_algo[algo2])
                    std1, std2 = np.std(rewards_by_algo[algo1]), np.std(rewards_by_algo[algo2])
                    
                    # Pooled standard deviation
                    n1, n2 = len(rewards_by_algo[algo1]), len(rewards_by_algo[algo2])
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                    
                    # Cohen's d
                    cohen_d = abs(mean1 - mean2) / pooled_std
                    
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
                
                # Performance improvement analysis
                f.write("#### Performance Improvement Analysis\n\n")
                
                # Find best performing algorithm
                avg_rewards = {algo: np.mean(rewards) for algo, rewards in rewards_by_algo.items() 
                              if "_normalized" not in algo and "_progress" not in algo}
                best_algo = max(avg_rewards, key=avg_rewards.get)
                
                f.write(f"Best performing algorithm (by mean reward): **{best_algo}**\n\n")
                
                # Calculate relative performance
                best_mean = avg_rewards[best_algo]
                
                f.write("| Algorithm | Mean Reward | % of Best |\n")
                f.write("|-----------|-------------|----------|\n")
                
                for algo, mean_reward in sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True):
                    pct_of_best = (mean_reward / best_mean) * 100
                    f.write(f"| {algo} | {mean_reward:.2f} | {pct_of_best:.1f}% |\n")
                
                f.write("\n")
                
                # Final performance comparison
                f.write("#### Final Performance Comparison\n\n")
                
                final_rewards = {}
                for algo, rewards in rewards_by_algo.items():
                    if "_normalized" not in algo and "_progress" not in algo:
                        final_rewards[algo] = rewards[-1]
                
                best_final = max(final_rewards.values())
                
                f.write("| Algorithm | Final Reward | % of Best |\n")
                f.write("|-----------|-------------|----------|\n")
                
                for algo, reward in sorted(final_rewards.items(), key=lambda x: x[1], reverse=True):
                    pct_of_best = (reward / best_final) * 100
                    f.write(f"| {algo} | {reward:.2f} | {pct_of_best:.1f}% |\n")
                
                f.write("\n")
        
        # Sample efficiency analysis
        if 'eval_rewards' in metrics_data:
            f.write("## Sample Efficiency Analysis\n\n")
            
            eval_rewards = metrics_data['eval_rewards'].copy()
            
            # Get the maximum reward achieved by any algorithm
            max_reward = eval_rewards['reward'].max()
            
            # Define performance thresholds
            thresholds = [0.25, 0.5, 0.75, 0.9]
            threshold_values = [max_reward * t for t in thresholds]
            
            f.write("### Steps to Reach Performance Thresholds\n\n")
            
            # Analyze both in raw steps and normalized progress
            for metric_type, header in [("steps", "Raw Steps"), ("normalized_steps", "Training Progress")]:
                if metric_type == "normalized_steps" and (ppo_steps == 0 or sac_steps == 0 or ppo_steps == sac_steps):
                    continue  # Skip normalized analysis if step counts are the same
                
                f.write(f"#### {header}\n\n")
                
                f.write("| Algorithm |")
                for t in thresholds:
                    f.write(f" {int(t*100)}% of Max |")
                f.write("\n")
                
                f.write("|-----------|")
                for _ in thresholds:
                    f.write("-------------|")
                f.write("\n")
                
                # Calculate steps needed for each algorithm to reach thresholds
                for algo in eval_rewards['algorithm'].unique():
                    algo_data = eval_rewards[eval_rewards['algorithm'] == algo].sort_values('steps')
                
                # Calculate CV for full training
                mean_reward = algo_data['reward'].mean()
                std_reward = algo_data['reward'].std()
                cv_full = (std_reward / mean_reward) if mean_reward != 0 else float('inf')
                
                # Calculate CV for last half of training
                half_point = len(algo_data) // 2
                later_data = algo_data.iloc[half_point:]
                
                mean_later = later_data['reward'].mean()
                std_later = later_data['reward'].std()
                cv_later = (std_later / mean_later) if mean_later != 0 else float('inf')
                
                f.write(f"| {algo} | {cv_full:.4f} | {cv_later:.4f} |\n")
            
            f.write("\n")
            f.write("*Note: Lower Coefficient of Variation (CV) indicates more stable training.*\n\n")
            
            # Add extra stability metrics considering the different step counts
            if ppo_steps != sac_steps and ppo_steps > 0 and sac_steps > 0:
                f.write("### Stability During Equivalent Training Phases\n\n")
                f.write("To fairly compare stability between algorithms with different training durations, ")
                f.write("we analyze the Coefficient of Variation during equivalent normalized training phases.\n\n")
                
                # Calculate CV for various training phases (0-25%, 25-50%, 50-75%, 75-100%)
                training_phases = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
                phase_names = ["Start (0-25%)", "Early (25-50%)", "Mid (50-75%)", "Late (75-100%)"]
                
                f.write("| Algorithm |")
                for phase in phase_names:
                    f.write(f" {phase} |")
                f.write("\n")
                
                f.write("|-----------|")
                for _ in phase_names:
                    f.write("-----------|")
                f.write("\n")
                
                for algo in eval_rewards['algorithm'].unique():
                    algo_data = eval_rewards[eval_rewards['algorithm'] == algo].sort_values('normalized_steps')
                    
                    f.write(f"| {algo} |")
                    
                    for start, end in training_phases:
                        # Get data for this phase
                        phase_data = algo_data[(algo_data['normalized_steps'] >= start) & 
                                             (algo_data['normalized_steps'] < end)]
                        
                        if len(phase_data) > 1:
                            mean_phase = phase_data['reward'].mean()
                            std_phase = phase_data['reward'].std()
                            cv_phase = (std_phase / mean_phase) if mean_phase != 0 else float('inf')
                            f.write(f" {cv_phase:.4f} |")
                        else:
                            f.write(" N/A |")
                    
                    f.write("\n")
                
                f.write("\n")
                
                f.write("### Stability Analysis Interpretation\n\n")
                f.write("- Training stability is critical for reliable and predictable learning in RL algorithms.\n")
                f.write("- Lower CV indicates less variance relative to the mean reward, suggesting more stable training.\n")
                f.write("- Typically, we expect stability to improve in later phases of training as policies converge.\n")
                
                # Attempt to identify more stable algorithm
                cv_by_algo = {}
                for algo in eval_rewards['algorithm'].unique():
                    algo_data = eval_rewards[eval_rewards['algorithm'] == algo].sort_values('steps')
                    mean_reward = algo_data['reward'].mean()
                    std_reward = algo_data['reward'].std()
                    cv_by_algo[algo] = (std_reward / mean_reward) if mean_reward != 0 else float('inf')
                
                if len(cv_by_algo) > 1:
                    most_stable = min(cv_by_algo, key=cv_by_algo.get)
                    f.write(f"\nOverall, **{most_stable}** showed more stable training behavior ")
                    f.write(f"with a coefficient of variation of {cv_by_algo[most_stable]:.4f}.\n")
            
        # Final conclusions
        f.write("\n## Overall Analysis\n\n")
        
        # Attempt to summarize key findings if we have both PPO and SAC metrics
        if 'ppo_eval_rewards' in metrics_data and 'sac_eval_rewards' in metrics_data:
            # Get final rewards
            ppo_final = metrics_data['ppo_eval_rewards'].sort_values('steps').iloc[-1]
            sac_final = metrics_data['sac_eval_rewards'].sort_values('steps').iloc[-1]
            
            # Get mean rewards
            ppo_mean = metrics_data['ppo_eval_rewards']['reward'].mean()
            sac_mean = metrics_data['sac_eval_rewards']['reward'].mean()
            
            # Step counts
            ppo_step_count = ppo_steps if ppo_steps > 0 else "unknown"
            sac_step_count = sac_steps if sac_steps > 0 else "unknown"
            
            f.write(f"### Key Observations\n\n")
            
            # Final performance comparison
            f.write(f"- **Final Performance**: ")
            if ppo_final['reward'] > sac_final['reward']:
                f.write(f"PPO achieved higher final reward ({ppo_final['reward']:.2f}) compared to ")
                f.write(f"SAC ({sac_final['reward']:.2f}), a difference of {(ppo_final['reward'] - sac_final['reward']):.2f} ")
                f.write(f"({(ppo_final['reward'] / sac_final['reward'] * 100 - 100):.1f}% higher).\n")
            elif sac_final['reward'] > ppo_final['reward']:
                f.write(f"SAC achieved higher final reward ({sac_final['reward']:.2f}) compared to ")
                f.write(f"PPO ({ppo_final['reward']:.2f}), a difference of {(sac_final['reward'] - ppo_final['reward']):.2f} ")
                f.write(f"({(sac_final['reward'] / ppo_final['reward'] * 100 - 100):.1f}% higher).\n")
            else:
                f.write(f"Both algorithms achieved similar final rewards around {ppo_final['reward']:.2f}.\n")
            
            # Training efficiency
            f.write(f"- **Training Duration**: PPO was trained for {ppo_step_count:,} steps, ")
            f.write(f"while SAC was trained for {sac_step_count:,} steps ")
            
            if ppo_steps > 0 and sac_steps > 0:
                if ppo_steps > sac_steps:
                    f.write(f"({ppo_steps / sac_steps:.1f}x more for PPO).\n")
                elif sac_steps > ppo_steps:
                    f.write(f"({sac_steps / ppo_steps:.1f}x more for SAC).\n")
                else:
                    f.write("(equal training duration).\n")
            else:
                f.write(".\n")
                
            # Sample efficiency comparison if we have data on steps to 50% performance
            if 'eval_rewards' in metrics_data:
                eval_rewards = metrics_data['eval_rewards'].copy()
                max_reward = eval_rewards['reward'].max()
                mid_threshold = max_reward * 0.5
                
                steps_to_mid = {}
                normalized_to_mid = {}
                
                for algo in eval_rewards['algorithm'].unique():
                    algo_data = eval_rewards[eval_rewards['algorithm'] == algo].sort_values('steps')
                    above_threshold = algo_data[algo_data['reward'] >= mid_threshold]
                    
                    if not above_threshold.empty:
                        steps_to_mid[algo] = above_threshold.iloc[0]['steps']
                        if 'normalized_steps' in above_threshold.columns:
                            normalized_to_mid[algo] = above_threshold.iloc[0]['normalized_steps']
                
                if len(steps_to_mid) > 1:
                    # Get algorithm names without step counts
                    ppo_algo = [a for a in steps_to_mid.keys() if "PPO" in a][0]
                    sac_algo = [a for a in steps_to_mid.keys() if "SAC" in a][0]
                    
                    # Compare raw steps
                    f.write(f"- **Sample Efficiency**: ")
                    if steps_to_mid[ppo_algo] < steps_to_mid[sac_algo]:
                        f.write(f"PPO reached 50% performance in {steps_to_mid[ppo_algo]:,} steps vs ")
                        f.write(f"SAC's {steps_to_mid[sac_algo]:,} steps ")
                        f.write(f"({steps_to_mid[sac_algo] / steps_to_mid[ppo_algo]:.1f}x faster for PPO).\n")
                    elif steps_to_mid[sac_algo] < steps_to_mid[ppo_algo]:
                        f.write(f"SAC reached 50% performance in {steps_to_mid[sac_algo]:,} steps vs ")
                        f.write(f"PPO's {steps_to_mid[ppo_algo]:,} steps ")
                        f.write(f"({steps_to_mid[ppo_algo] / steps_to_mid[sac_algo]:.1f}x faster for SAC).\n")
                    else:
                        f.write(f"Both algorithms reached 50% performance in similar step counts.\n")
                    
                    # If we have normalized data, compare that as well
                    if len(normalized_to_mid) > 1:
                        f.write(f"- **Normalized Efficiency**: ")
                        if normalized_to_mid[ppo_algo] < normalized_to_mid[sac_algo]:
                            f.write(f"PPO reached 50% performance after completing {normalized_to_mid[ppo_algo]*100:.1f}% of training vs ")
                            f.write(f"SAC's {normalized_to_mid[sac_algo]*100:.1f}% ")
                            f.write(f"(relatively faster progress for PPO).\n")
                        elif normalized_to_mid[sac_algo] < normalized_to_mid[ppo_algo]:
                            f.write(f"SAC reached 50% performance after completing {normalized_to_mid[sac_algo]*100:.1f}% of training vs ")
                            f.write(f"PPO's {normalized_to_mid[ppo_algo]*100:.1f}% ")
                            f.write(f"(relatively faster progress for SAC).\n")
                        else:
                            f.write(f"Both algorithms showed similar relative progress rates.\n")
            
            # Stability comparison
            if 'ppo_eval_rewards' in metrics_data and 'sac_eval_rewards' in metrics_data:
                ppo_data = metrics_data['ppo_eval_rewards']
                sac_data = metrics_data['sac_eval_rewards']
                
                ppo_cv = ppo_data['reward'].std() / ppo_data['reward'].mean() if ppo_data['reward'].mean() != 0 else float('inf')
                sac_cv = sac_data['reward'].std() / sac_data['reward'].mean() if sac_data['reward'].mean() != 0 else float('inf')
                
                f.write(f"- **Training Stability**: ")
                if ppo_cv < sac_cv:
                    f.write(f"PPO showed more stable training (CV: {ppo_cv:.4f}) compared to ")
                    f.write(f"SAC (CV: {sac_cv:.4f}).\n")
                elif sac_cv < ppo_cv:
                    f.write(f"SAC showed more stable training (CV: {sac_cv:.4f}) compared to ")
                    f.write(f"PPO (CV: {ppo_cv:.4f}).\n")
                else:
                    f.write(f"Both algorithms showed similar stability in training.\n")
            
            # Overall conclusion
            f.write("\n### Conclusion\n\n")
            
            # Attempt to give a balanced conclusion based on the data
            if ppo_final['reward'] > sac_final['reward'] and ppo_steps > sac_steps:
                f.write("PPO achieved higher final performance but required significantly more training steps. ")
                if 'normalized_to_mid' in locals() and len(normalized_to_mid) > 1:
                    ppo_algo = [a for a in normalized_to_mid.keys() if "PPO" in a][0]
                    sac_algo = [a for a in normalized_to_mid.keys() if "SAC" in a][0]
                    if normalized_to_mid[sac_algo] < normalized_to_mid[ppo_algo]:
                        f.write("SAC showed faster relative progress during training, suggesting better sample efficiency. ")
                    else:
                        f.write("PPO showed faster relative progress despite needing more total steps. ")
                        
                if ppo_cv < sac_cv:
                    f.write("PPO training was more stable, showing less variation in rewards. ")
                else:
                    f.write("SAC training was more stable despite achieving lower final performance. ")
                    
                f.write("\n\nThe choice between these algorithms depends on specific requirements:\n")
                f.write("- If computational resources allow for longer training and maximum performance is the priority, PPO may be preferred.\n")
                f.write("- If sample efficiency and quicker deployment are important, SAC could be the better option despite potentially lower final performance.\n")
                
            elif sac_final['reward'] > ppo_final['reward'] and ppo_steps > sac_steps:
                f.write("SAC achieved higher final performance while requiring fewer training steps. ")
                f.write("This suggests SAC is both more effective and more sample efficient for this task. ")
                
                if ppo_cv < sac_cv:
                    f.write("However, PPO training was more stable, showing less variation in rewards. ")
                else:
                    f.write("SAC also demonstrated more stable training behavior. ")
                    
                f.write("\n\nBased on these results, SAC appears to be the superior algorithm for this particular environment, ")
                f.write("offering better performance with fewer computational resources. The only potential advantage of PPO ")
                f.write("might be in environments where training stability is the primary concern.\n")
                
            elif ppo_final['reward'] > sac_final['reward'] and sac_steps >= ppo_steps:
                f.write("PPO achieved higher final performance with equal or fewer training steps compared to SAC. ")
                f.write("This suggests PPO is both more effective and more sample efficient for this specific task. ")
                
                if ppo_cv < sac_cv:
                    f.write("PPO training was also more stable, showing less variation in rewards. ")
                else:
                    f.write("However, SAC demonstrated more stable training behavior. ")
                    
                f.write("\n\nBased on these results, PPO appears to be the superior algorithm for this particular environment. ")
                f.write("It achieves better performance without requiring more resources than SAC.\n")
                
            elif sac_final['reward'] > ppo_final['reward'] and sac_steps >= ppo_steps:
                f.write("SAC achieved higher final performance despite requiring equal or more training steps. ")
                
                if sac_cv < ppo_cv:
                    f.write("SAC training was also more stable, showing less variation in rewards. ")
                else:
                    f.write("However, PPO demonstrated more stable training behavior. ")
                    
                f.write("\n\nSAC appears to be the more effective algorithm for this task, though it may be less sample efficient. ")
                f.write("If computational resources are limited, further investigation might be needed to find the optimal ")
                f.write("training duration for both algorithms.\n")
                
            else:
                f.write("Both algorithms performed comparably in terms of final performance. ")
                
                if ppo_steps != sac_steps:
                    if ppo_steps > sac_steps:
                        f.write("SAC achieved this with fewer training steps, suggesting better sample efficiency. ")
                    else:
                        f.write("PPO achieved this with fewer training steps, suggesting better sample efficiency. ")
                        
                if ppo_cv != sac_cv:
                    if ppo_cv < sac_cv:
                        f.write("PPO showed more stable training behavior with less reward variance. ")
                    else:
                        f.write("SAC showed more stable training behavior with less reward variance. ")
                        
                f.write("\n\nThe choice between these algorithms may depend on factors beyond performance, ")
                f.write("such as implementation complexity, hyperparameter sensitivity, or specific requirements ")
                f.write("of the deployment environment.\n")

def main():
    """Main function for analyzing results."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {args.results_dir}...")
    ppo_metrics, sac_metrics, comparison_df, config = load_metrics(args.results_dir)
    
    # Prepare data for analysis
    print("Preparing data for analysis...")
    metrics_data = prepare_data(ppo_metrics, sac_metrics, config)
    
    # Plot learning curves and other visualizations
    print("Generating plots...")
    plot_learning_curves(metrics_data, args, args.output_dir)
    
    # Generate statistical analysis
    print("Performing statistical analysis...")
    generate_statistical_analysis(metrics_data, config, args.output_dir)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()