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
    
    return parser.parse_args()

def load_metrics(results_dir):
    """Load metrics from JSON files."""
    
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

def prepare_data(ppo_metrics, sac_metrics):
    """Prepare data for analysis."""
    
    metrics_data = {}
    
    # Create DataFrames for each metric type
    if ppo_metrics:
        # Evaluation rewards
        ppo_eval_rewards = pd.DataFrame(ppo_metrics['eval_mean_rewards'], 
                                       columns=['steps', 'reward'])
        ppo_eval_rewards['algorithm'] = 'PPO-CLIP'
        ppo_eval_rewards['std'] = [std for _, std in ppo_metrics['eval_std_rewards']]
        
        # Training rewards - need to potentially aggregate by step
        ppo_train_rewards = pd.DataFrame(ppo_metrics['train_rewards'],
                                        columns=['steps', 'reward'])
        ppo_train_rewards['algorithm'] = 'PPO-CLIP'
        
        # Losses
        ppo_value_losses = pd.DataFrame(ppo_metrics['value_losses'],
                                       columns=['steps', 'loss'])
        ppo_value_losses['type'] = 'value_loss'
        ppo_value_losses['algorithm'] = 'PPO-CLIP'
        
        ppo_policy_losses = pd.DataFrame(ppo_metrics['policy_losses'],
                                        columns=['steps', 'loss'])
        ppo_policy_losses['type'] = 'policy_loss'
        ppo_policy_losses['algorithm'] = 'PPO-CLIP'
        
        ppo_entropy_losses = pd.DataFrame(ppo_metrics['entropy_losses'],
                                         columns=['steps', 'loss'])
        ppo_entropy_losses['type'] = 'entropy_loss'
        ppo_entropy_losses['algorithm'] = 'PPO-CLIP'
        
        # Combine losses
        ppo_losses = pd.concat([ppo_value_losses, ppo_policy_losses, ppo_entropy_losses])
        
        metrics_data['ppo_eval_rewards'] = ppo_eval_rewards
        metrics_data['ppo_train_rewards'] = ppo_train_rewards
        metrics_data['ppo_losses'] = ppo_losses
    
    if sac_metrics:
        # Evaluation rewards
        sac_eval_rewards = pd.DataFrame(sac_metrics['eval_mean_rewards'], 
                                       columns=['steps', 'reward'])
        sac_eval_rewards['algorithm'] = 'SAC'
        sac_eval_rewards['std'] = [std for _, std in sac_metrics['eval_std_rewards']]
        
        # Training rewards
        sac_train_rewards = pd.DataFrame(sac_metrics['train_rewards'],
                                        columns=['steps', 'reward'])
        sac_train_rewards['algorithm'] = 'SAC'
        
        # Losses
        sac_critic1_losses = pd.DataFrame(sac_metrics['critic_1_losses'],
                                         columns=['steps', 'loss'])
        sac_critic1_losses['type'] = 'critic_1_loss'
        sac_critic1_losses['algorithm'] = 'SAC'
        
        sac_critic2_losses = pd.DataFrame(sac_metrics['critic_2_losses'],
                                         columns=['steps', 'loss'])
        sac_critic2_losses['type'] = 'critic_2_loss'
        sac_critic2_losses['algorithm'] = 'SAC'
        
        sac_policy_losses = pd.DataFrame(sac_metrics['policy_losses'],
                                        columns=['steps', 'loss'])
        sac_policy_losses['type'] = 'policy_loss'
        sac_policy_losses['algorithm'] = 'SAC'
        
        sac_alpha_losses = pd.DataFrame(sac_metrics['alpha_losses'],
                                      columns=['steps', 'loss'])
        sac_alpha_losses['type'] = 'alpha_loss'
        sac_alpha_losses['algorithm'] = 'SAC'
        
        # Combine losses
        sac_losses = pd.concat([sac_critic1_losses, sac_critic2_losses, 
                              sac_policy_losses, sac_alpha_losses])
        
        # Alpha values
        sac_alphas = pd.DataFrame(sac_metrics['alphas'],
                                 columns=['steps', 'alpha'])
        
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
    
    return metrics_data

def apply_smoothing(data, window_size=5):
    """Apply moving average smoothing to data."""
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_learning_curves(metrics_data, args, output_dir):
    """Plot learning curves comparison."""
    
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
    
    # 1. Plot evaluation rewards with confidence intervals
    if 'eval_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # Group by algorithm and steps
        grouped = metrics_data['eval_rewards'].groupby(['algorithm', 'steps']).agg({
            'reward': 'mean', 
            'std': 'mean'
        }).reset_index()
        
        # Apply smoothing
        for algo in grouped['algorithm'].unique():
            algo_data = grouped[grouped['algorithm'] == algo]
            
            # Sort by steps
            algo_data = algo_data.sort_values('steps')
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_rewards = apply_smoothing(algo_data['reward'], args.smoothing)
                smoothed_stds = apply_smoothing(algo_data['std'], args.smoothing)
            else:
                smoothed_rewards = algo_data['reward']
                smoothed_stds = algo_data['std']
            
            # Plot mean
            line = ax.plot(algo_data['steps'], smoothed_rewards, 
                          label=algo, linewidth=2)
            
            # Add confidence interval
            if not args.no_confidence:
                ax.fill_between(
                    algo_data['steps'],
                    smoothed_rewards - smoothed_stds,
                    smoothed_rewards + smoothed_stds,
                    alpha=0.2,
                    color=line[0].get_color()
                )
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Mean Evaluation Reward')
        ax.set_title('Evaluation Rewards Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for maximum performance
        for algo in grouped['algorithm'].unique():
            algo_data = grouped[grouped['algorithm'] == algo]
            max_idx = algo_data['reward'].idxmax()
            max_step = algo_data.loc[max_idx, 'steps']
            max_reward = algo_data.loc[max_idx, 'reward']
            
            ax.annotate(f'Max: {max_reward:.2f}',
                      xy=(max_step, max_reward),
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
        
        # Determine bin size based on total steps
        max_steps = train_rewards['steps'].max()
        bin_size = max(1, max_steps // 100)  # About 100 points on the x-axis
        
        # Create step bins
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
            line = ax.plot(algo_data['step_bin'], smoothed_rewards, 
                          label=algo, linewidth=2, alpha=0.8)
            
            # Add confidence interval
            if not args.no_confidence and smoothed_stds is not None:
                ax.fill_between(
                    algo_data['step_bin'],
                    smoothed_rewards - smoothed_stds,
                    smoothed_rewards + smoothed_stds,
                    alpha=0.1,
                    color=line[0].get_color()
                )
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Mean Training Episode Reward')
        ax.set_title('Training Rewards Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_rewards_comparison.png'))
        plt.close()
    
    # 3. Plot loss curves for each algorithm
    # PPO Losses
    if 'ppo_losses' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        ppo_losses = metrics_data['ppo_losses'].copy()
        
        # Bin losses by steps
        max_steps = ppo_losses['steps'].max()
        bin_size = max(1, max_steps // 100)
        ppo_losses['step_bin'] = (ppo_losses['steps'] // bin_size) * bin_size
        
        # Group by loss type and step bin
        grouped = ppo_losses.groupby(['type', 'step_bin']).agg({
            'loss': 'mean'
        }).reset_index()
        
        # Plot each loss type
        for loss_type in grouped['type'].unique():
            type_data = grouped[grouped['type'] == loss_type]
            
            # Sort by step bin
            type_data = type_data.sort_values('step_bin')
            
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
                ax.plot(type_data['step_bin'], scaled_losses, 
                       label=f'{loss_type} (scaled)', linewidth=2)
            else:
                ax.plot(type_data['step_bin'], smoothed_losses, 
                       label=loss_type, linewidth=2)
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Loss')
        ax.set_title('PPO Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ppo_losses.png'))
        plt.close()
    
    # SAC Losses
    if 'sac_losses' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        sac_losses = metrics_data['sac_losses'].copy()
        
        # Bin losses by steps
        max_steps = sac_losses['steps'].max()
        bin_size = max(1, max_steps // 100)
        sac_losses['step_bin'] = (sac_losses['steps'] // bin_size) * bin_size
        
        # Group by loss type and step bin
        grouped = sac_losses.groupby(['type', 'step_bin']).agg({
            'loss': 'mean'
        }).reset_index()
        
        # Plot each loss type
        for loss_type in grouped['type'].unique():
            type_data = grouped[grouped['type'] == loss_type]
            
            # Sort by step bin
            type_data = type_data.sort_values('step_bin')
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_losses = apply_smoothing(type_data['loss'], args.smoothing)
            else:
                smoothed_losses = type_data['loss']
            
            # Plot
            ax.plot(type_data['step_bin'], smoothed_losses, 
                   label=loss_type, linewidth=2)
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Loss')
        ax.set_title('SAC Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sac_losses.png'))
        plt.close()
    
    # 4. Plot SAC alpha parameter
    if 'sac_alphas' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        sac_alphas = metrics_data['sac_alphas'].copy()
        
        # Sort by steps
        sac_alphas = sac_alphas.sort_values('steps')
        
        # Apply smoothing
        if args.smoothing > 1:
            smoothed_alphas = apply_smoothing(sac_alphas['alpha'], args.smoothing)
        else:
            smoothed_alphas = sac_alphas['alpha']
        
        # Plot
        ax.plot(sac_alphas['steps'], smoothed_alphas, 
               label='alpha', linewidth=2)
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Alpha Value')
        ax.set_title('SAC Temperature Parameter (Alpha)')
        ax.grid(True, alpha=0.3)
        
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
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_performance.png'))
        plt.close()
    
    # 6. Learning progress visualization
    if 'eval_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        eval_rewards = metrics_data['eval_rewards'].copy()
        
        # Get maximum reward value overall
        max_reward = eval_rewards['reward'].max()
        
        # For each algorithm, calculate percentage of max reward achieved over time
        for algo in eval_rewards['algorithm'].unique():
            algo_data = eval_rewards[eval_rewards['algorithm'] == algo]
            
            # Sort by steps
            algo_data = algo_data.sort_values('steps')
            
            # Calculate percentage of max reward
            algo_data['progress'] = algo_data['reward'] / max_reward * 100
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_progress = apply_smoothing(algo_data['progress'], args.smoothing)
            else:
                smoothed_progress = algo_data['progress']
            
            # Plot
            ax.plot(algo_data['steps'], smoothed_progress, 
                   label=algo, linewidth=2)
        
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
    
    # 7. Sample efficiency comparison
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
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f"{int(value):,}",
                           ha='center', va='bottom',
                           rotation=45, fontsize=args.font_size-2)
        
        # Set x-axis labels and ticks
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds])
        
        ax.set_xlabel('Performance Threshold (% of max reward)')
        ax.set_ylabel('Steps to Reach Threshold')
        ax.set_title('Sample Efficiency Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_efficiency.png'))
        plt.close()
    
    # 8. Performance stability analysis
    if 'eval_rewards' in metrics_data:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        eval_rewards = metrics_data['eval_rewards'].copy()
        
        # Calculate moving average and standard deviation for stability analysis
        window_size = max(5, int(len(eval_rewards) / 10))  # Dynamic window size
        
        for algo in eval_rewards['algorithm'].unique():
            algo_data = eval_rewards[eval_rewards['algorithm'] == algo].sort_values('steps')
            
            # Calculate rolling metrics
            if len(algo_data) > window_size:
                rolling_mean = algo_data['reward'].rolling(window=window_size, min_periods=1).mean()
                rolling_std = algo_data['reward'].rolling(window=window_size, min_periods=1).std()
                
                # Calculate coefficient of variation (CV = std/mean) as stability metric
                # Higher CV = more variability = less stable
                cv = rolling_std / rolling_mean
                
                # Plot stability metric
                ax.plot(algo_data['steps'], cv, 
                       label=f"{algo} Variability", linewidth=2)
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Coefficient of Variation (Lower = More Stable)')
        ax.set_title('Learning Stability Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stability_analysis.png'))
        plt.close()

    # 9. Training efficiency comparison
    if ppo_metrics and sac_metrics and 'times' in ppo_metrics and 'times' in sac_metrics:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # Extract timing data
        ppo_steps = np.array(ppo_metrics['steps'])
        ppo_times = np.array(ppo_metrics['times'])
        
        sac_steps = np.array(sac_metrics['steps'])
        sac_times = np.array(sac_metrics['times'])
        
        # Calculate steps per second
        ppo_steps_per_sec = ppo_steps / ppo_times
        sac_steps_per_sec = sac_steps / sac_times
        
        # Create DataFrame for plotting
        efficiency_data = pd.DataFrame({
            'Algorithm': ['PPO-CLIP'] * len(ppo_steps) + ['SAC'] * len(sac_steps),
            'Step': list(ppo_steps) + list(sac_steps),
            'Time': list(ppo_times) + list(sac_times),
            'Steps/Second': list(ppo_steps_per_sec) + list(sac_steps_per_sec)
        })
        
        # Plot as violin plot
        sns.violinplot(x='Algorithm', y='Steps/Second', data=efficiency_data, ax=ax)
        
        # Add individual points
        sns.stripplot(x='Algorithm', y='Steps/Second', data=efficiency_data, 
                     color='black', size=3, jitter=True, alpha=0.3, ax=ax)
        
        # Add average values as text
        for i, algo in enumerate(['PPO-CLIP', 'SAC']):
            avg_speed = efficiency_data[efficiency_data['Algorithm'] == algo]['Steps/Second'].mean()
            ax.text(i, efficiency_data['Steps/Second'].max() * 0.9, 
                   f"Avg: {avg_speed:.1f}\nsteps/sec", 
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_ylabel('Environment Steps per Second')
        ax.set_title('Training Efficiency Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_efficiency.png'))
        plt.close()

def generate_statistical_analysis(metrics_data, output_dir):
    """Generate statistical analysis of the results."""
    
    # Create a text file to store the analysis
    stats_file = os.path.join(output_dir, 'statistical_analysis.txt')
    
    with open(stats_file, 'w') as f:
        f.write("# Statistical Analysis of Algorithm Performance\n\n")
        
        if 'eval_rewards' in metrics_data:
            f.write("## Evaluation Rewards Analysis\n\n")
            
            # Group evaluation rewards by algorithm
            algorithms = metrics_data['eval_rewards']['algorithm'].unique()
            
            rewards_by_algo = {}
            for algo in algorithms:
                algo_data = metrics_data['eval_rewards'][metrics_data['eval_rewards']['algorithm'] == algo]
                rewards_by_algo[algo] = algo_data['reward'].values
            
            # Basic statistics for each algorithm
            f.write("### Basic Statistics\n\n")
            f.write("| Algorithm | Mean | Median | Std Dev | Min | Max | Final |\n")
            f.write("|-----------|------|--------|---------|-----|-----|-------|\n")
            
            for algo, rewards in rewards_by_algo.items():
                last_reward = rewards[-1]
                f.write(f"| {algo} | {np.mean(rewards):.2f} | {np.median(rewards):.2f} | "
                       f"{np.std(rewards):.2f} | {np.min(rewards):.2f} | {np.max(rewards):.2f} | "
                       f"{last_reward:.2f} |\n")
            
            f.write("\n")
            
            # Comparative statistics if we have multiple algorithms
            if len(algorithms) > 1:
                f.write("### Comparative Analysis\n\n")
                
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
                avg_rewards = {algo: np.mean(rewards) for algo, rewards in rewards_by_algo.items()}
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
                
                f.write(f"| {algo} |")
                
                for threshold_value in threshold_values:
                    above_threshold = algo_data[algo_data['reward'] >= threshold_value]
                    
                    if not above_threshold.empty:
                        first_crossing = above_threshold.iloc[0]
                        f.write(f" {int(first_crossing['steps']):,} |")
                    else:
                        f.write(" Never |")
                
                f.write("\n")
            
            f.write("\n")
            
            # Add interpretation
            f.write("### Interpretation\n\n")
            f.write("Sample efficiency refers to how quickly an algorithm learns from experience. ")
            f.write("An algorithm is more sample efficient if it reaches performance thresholds ")
            f.write("with fewer environment interactions (steps).\n\n")
            
            # Try to identify which algorithm is more sample efficient
            if len(eval_rewards['algorithm'].unique()) > 1:
                # Compare sample efficiency at the middle threshold (50%)
                mid_threshold = threshold_values[1]  # 50% threshold
                
                steps_to_mid = {}
                for algo in eval_rewards['algorithm'].unique():
                    algo_data = eval_rewards[eval_rewards['algorithm'] == algo].sort_values('steps')
                    above_threshold = algo_data[algo_data['reward'] >= mid_threshold]
                    
                    if not above_threshold.empty:
                        steps_to_mid[algo] = above_threshold.iloc[0]['steps']
                
                if len(steps_to_mid) > 1:
                    most_efficient = min(steps_to_mid, key=steps_to_mid.get)
                    f.write(f"Based on the steps needed to reach 50% of maximum performance, ")
                    f.write(f"**{most_efficient}** appears to be more sample efficient.\n\n")
        
        # Training stability analysis
        if 'eval_rewards' in metrics_data:
            f.write("## Training Stability Analysis\n\n")
            
            eval_rewards = metrics_data['eval_rewards'].copy()
            
            # Calculate coefficient of variation for each algorithm
            f.write("| Algorithm | CV (Full Training) | CV (Last Half) |\n")
            f.write("|-----------|---------------------|----------------|\n")
            
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
    metrics_data = prepare_data(ppo_metrics, sac_metrics)
    
    # Plot learning curves and other visualizations
    print("Generating plots...")
    plot_learning_curves(metrics_data, args, args.output_dir)
    
    # Generate statistical analysis
    print("Performing statistical analysis...")
    generate_statistical_analysis(metrics_data, args.output_dir)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()