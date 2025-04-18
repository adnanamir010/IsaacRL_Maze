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
    """Parse command line arguments for the SAC variants analysis script."""
    parser = argparse.ArgumentParser(description='Analyze SAC variants comparison results')
    
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
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum steps to include in analysis (default: None)')
    
    return parser.parse_args()

def load_tensorboard_data(logs_dir):
    """Load data from TensorBoard logs."""
    print(f"Loading TensorBoard data from {logs_dir}...")
    
    # Pattern to identify variant folders
    pattern = r"(Single|Twin) Critic \+ (Adaptive|Fixed|No) Entropy_"
    
    # Find all variant directories
    variant_dirs = [d for d in os.listdir(logs_dir) if re.search(pattern, d)]
    
    if not variant_dirs:
        print("Error: No variant directories found. Make sure the directory structure matches the expected pattern.")
        return None
    
    print(f"Found {len(variant_dirs)} variant directories: {variant_dirs}")
    
    # Data structure to store metrics
    metrics_data = {}
    
    # Metrics to extract
    metrics_of_interest = [
        'eval/mean_reward',
        'eval/std_reward',
        'train/alpha',
        'train/critic_1_loss',
        'train/critic_2_loss',
        'train/entropy_loss',
        'train/episode_reward',
        'train/episode_steps',
        'train/policy_loss'
    ]
    
    # Extract variant info and load data
    for variant_dir in variant_dirs:
        # Parse variant details
        match = re.search(pattern, variant_dir)
        if match:
            critic_type = match.group(1)
            entropy_type = match.group(2)
            variant_name = f"{critic_type} Critic + {entropy_type} Entropy"
            
            print(f"Processing {variant_name}...")
            
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
            if variant_name not in metrics_data:
                metrics_data[variant_name] = {}
            
            # Extract metrics
            for tag in tags_to_load:
                events = ea.Scalars(tag)
                
                if not events:
                    print(f"Warning: No events found for {tag} in {variant_dir}")
                    continue
                
                # Extract step, value, and wall_time
                steps = [event.step for event in events]
                values = [event.value for event in events]
                
                # Store in the data structure
                clean_tag = tag.replace('/', '_')
                metrics_data[variant_name][clean_tag] = {
                    'steps': steps,
                    'values': values
                }
    
    return metrics_data

def prepare_dataframes(metrics_data):
    """Convert metrics data to pandas DataFrames for easier analysis."""
    dfs = {}
    
    # Create DataFrame for each metric type
    metrics_to_process = [
        'eval_mean_reward',
        'eval_std_reward',
        'train_episode_reward',
        'train_critic_1_loss',
        'train_critic_2_loss',
        'train_policy_loss',
        'train_entropy_loss',
        'train_alpha'
    ]
    
    for metric in metrics_to_process:
        data = []
        
        for variant_name, variant_data in metrics_data.items():
            if metric in variant_data:
                # Extract critic type and entropy type from variant name
                critic_type, entropy_type = variant_name.split(' + ')
                
                # Create rows for the DataFrame
                for step, value in zip(variant_data[metric]['steps'], variant_data[metric]['values']):
                    data.append({
                        'step': step,
                        'value': value,
                        'variant_name': variant_name,
                        'critic_type': critic_type,
                        'entropy_type': entropy_type
                    })
        
        if data:
            dfs[metric] = pd.DataFrame(data)
    
    return dfs

def apply_smoothing(data, window_size=5):
    """Apply moving average smoothing to data."""
    return data.rolling(window=window_size, min_periods=1).mean()

def plot_learning_curves(dfs, args, output_dir):
    """Generate learning curve plots for the different variants."""
    
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
    
    # Plot evaluation rewards with confidence intervals
    if 'eval_mean_reward' in dfs and 'eval_std_reward' in dfs:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # Merge the reward data with std data
        eval_rewards = dfs['eval_mean_reward'].copy()
        std_rewards = dfs['eval_std_reward'].copy()
        
        # Create a mapping from (variant_name, step) to std
        std_map = {}
        for _, row in std_rewards.iterrows():
            std_map[(row['variant_name'], row['step'])] = row['value']
        
        # Add std to the eval_rewards DataFrame
        eval_rewards['std'] = eval_rewards.apply(
            lambda row: std_map.get((row['variant_name'], row['step']), 0), 
            axis=1
        )
        
        # Colors for different variants
        colors = {
            'Twin Critic + Adaptive Entropy': 'blue',
            'Twin Critic + Fixed Entropy': 'green',
            'Twin Critic + No Entropy': 'red',
            'Single Critic + Adaptive Entropy': 'purple',
            'Single Critic + Fixed Entropy': 'orange',
            'Single Critic + No Entropy': 'brown'
        }
        
        # Line styles for entropy types
        line_styles = {
            'Adaptive Entropy': '-',
            'Fixed Entropy': '--',
            'No Entropy': ':'
        }
        
        # Plot each variant's learning curve
        for variant_name in eval_rewards['variant_name'].unique():
            variant_data = eval_rewards[eval_rewards['variant_name'] == variant_name].copy()
            
            # Extract entropy type from variant name
            entropy_type = variant_name.split(' + ')[1]
            
            # Sort by steps
            variant_data = variant_data.sort_values('step')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                variant_data = variant_data[variant_data['step'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                # Group by step to ensure we have unique step values
                grouped = variant_data.groupby('step').agg({
                    'value': 'mean',
                    'std': 'mean'
                }).reset_index()
                
                # Sort and apply smoothing
                grouped = grouped.sort_values('step')
                smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                smoothed_stds = apply_smoothing(grouped['std'], args.smoothing)
            else:
                grouped = variant_data.sort_values('step')
                smoothed_rewards = grouped['value']
                smoothed_stds = grouped['std']
            
            # Plot mean
            color = colors.get(variant_name, 'gray')
            linestyle = line_styles.get(entropy_type, '-')
            
            line = ax.plot(grouped['step'], smoothed_rewards,
                          label=variant_name, linewidth=2,
                          color=color, linestyle=linestyle)
            
            # Add confidence interval
            if not args.no_confidence:
                ax.fill_between(
                    grouped['step'],
                    smoothed_rewards - smoothed_stds,
                    smoothed_rewards + smoothed_stds,
                    alpha=0.2,
                    color=line[0].get_color()
                )
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Evaluation Reward')
        ax.set_title('SAC Variants: Evaluation Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for maximum performance
        for variant_name in eval_rewards['variant_name'].unique():
            variant_data = eval_rewards[eval_rewards['variant_name'] == variant_name]
            
            if args.max_steps:
                variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
            max_idx = variant_data['value'].idxmax()
            max_step = variant_data.loc[max_idx, 'step']
            max_reward = variant_data.loc[max_idx, 'value']
            
            # Add annotation with an arrow
            ax.annotate(f'{variant_name}: {max_reward:.2f}',
                      xy=(max_step, max_reward),
                      xytext=(max_step, max_reward + 15),
                      textcoords='data',
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eval_rewards_comparison.png'))
        plt.close()
    
    # Plot training rewards
    if 'train_episode_reward' in dfs:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        train_rewards = dfs['train_episode_reward'].copy()
        
        # Bin rewards by steps to make the plot cleaner
        bin_size = 5000  # Adjust bin size as needed
        train_rewards['step_bin'] = (train_rewards['step'] // bin_size) * bin_size
        
        # Group by variant and step bin
        grouped = train_rewards.groupby(['variant_name', 'step_bin']).agg({
            'value': ['mean', 'std']
        }).reset_index()
        
        # Flatten multi-index columns
        grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in grouped.columns]
        
        # Colors and line styles as above
        colors = {
            'Twin Critic + Adaptive Entropy': 'blue',
            'Twin Critic + Fixed Entropy': 'green',
            'Twin Critic + No Entropy': 'red',
            'Single Critic + Adaptive Entropy': 'purple',
            'Single Critic + Fixed Entropy': 'orange',
            'Single Critic + No Entropy': 'brown'
        }
        
        line_styles = {
            'Adaptive Entropy': '-',
            'Fixed Entropy': '--',
            'No Entropy': ':'
        }
        
        # Plot for each variant
        for variant_name in grouped['variant_name'].unique():
            variant_data = grouped[grouped['variant_name'] == variant_name]
            
            # Extract entropy type from variant name
            entropy_type = variant_name.split(' + ')[1]
            
            # Sort by step bin
            variant_data = variant_data.sort_values('step_bin')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                variant_data = variant_data[variant_data['step_bin'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_rewards = apply_smoothing(variant_data['value_mean'], args.smoothing)
                if 'value_std' in variant_data.columns:
                    smoothed_stds = apply_smoothing(variant_data['value_std'], args.smoothing)
                else:
                    smoothed_stds = None
            else:
                smoothed_rewards = variant_data['value_mean']
                smoothed_stds = variant_data['value_std'] if 'value_std' in variant_data.columns else None
            
            # Plot mean
            color = colors.get(variant_name, 'gray')
            linestyle = line_styles.get(entropy_type, '-')
            
            line = ax.plot(variant_data['step_bin'], smoothed_rewards, 
                          label=variant_name, linewidth=2, alpha=0.8,
                          color=color, linestyle=linestyle)
            
            # Add confidence interval
            if not args.no_confidence and smoothed_stds is not None:
                ax.fill_between(
                    variant_data['step_bin'],
                    smoothed_rewards - smoothed_stds,
                    smoothed_rewards + smoothed_stds,
                    alpha=0.1,
                    color=line[0].get_color()
                )
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Training Episode Reward')
        ax.set_title('SAC Variants: Training Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_rewards_comparison.png'))
        plt.close()
    
    # Plot Loss Components for each Variant
    loss_metrics = ['train_critic_1_loss', 'train_critic_2_loss', 'train_policy_loss', 'train_entropy_loss']
    available_loss_metrics = [m for m in loss_metrics if m in dfs]
    
    if available_loss_metrics:
        # Create a multi-panel figure for each variant
        for variant_name in dfs[available_loss_metrics[0]]['variant_name'].unique():
            fig, axes = plt.subplots(len(available_loss_metrics), 1, figsize=(10, 3*len(available_loss_metrics)), dpi=args.dpi)
            
            if len(available_loss_metrics) == 1:
                axes = [axes]  # Make it iterable if there's only one subplot
                
            for i, metric in enumerate(available_loss_metrics):
                metric_data = dfs[metric][dfs[metric]['variant_name'] == variant_name].copy()
                
                # Bin by steps
                bin_size = 5000  # Adjust as needed
                metric_data['step_bin'] = (metric_data['step'] // bin_size) * bin_size
                
                # Group by step bin
                grouped = metric_data.groupby('step_bin').agg({
                    'value': 'mean'
                }).reset_index()
                
                # Sort by step bin
                grouped = grouped.sort_values('step_bin')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    grouped = grouped[grouped['step_bin'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    smoothed_values = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    smoothed_values = grouped['value']
                
                # Plot
                metric_name = metric.split('_')[-2] + '_' + metric.split('_')[-1]
                axes[i].plot(grouped['step_bin'], smoothed_values, 
                            label=metric_name, linewidth=2)
                
                axes[i].set_xlabel('Training Steps')
                axes[i].set_ylabel(f'{metric_name.replace("_", " ").title()}')
                axes[i].set_title(f'{variant_name}: {metric_name.replace("_", " ").title()}')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            # Create a clean filename from the variant name
            variant_filename = variant_name.replace(' ', '_').replace('+', 'plus')
            plt.savefig(os.path.join(output_dir, f'{variant_filename}_losses.png'))
            plt.close()
    
    # Plot Alpha values for adaptive entropy variants
    if 'train_alpha' in dfs:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        alpha_data = dfs['train_alpha'].copy()
        
        # Only plot for variants with adaptive entropy
        adaptive_variants = [v for v in alpha_data['variant_name'].unique() 
                            if 'Adaptive' in v]
        
        for variant_name in adaptive_variants:
            variant_data = alpha_data[alpha_data['variant_name'] == variant_name].copy()
            
            # Bin by steps
            bin_size = 5000  # Adjust as needed
            variant_data['step_bin'] = (variant_data['step'] // bin_size) * bin_size
            
            # Group by step bin
            grouped = variant_data.groupby('step_bin').agg({
                'value': 'mean'
            }).reset_index()
            
            # Sort by step bin
            grouped = grouped.sort_values('step_bin')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                grouped = grouped[grouped['step_bin'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_values = apply_smoothing(grouped['value'], args.smoothing)
            else:
                smoothed_values = grouped['value']
            
            # Plot
            color = colors.get(variant_name, 'gray')
            ax.plot(grouped['step_bin'], smoothed_values, 
                   label=variant_name, linewidth=2, color=color)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Alpha Value')
        ax.set_title('SAC Temperature Parameter (Alpha)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alpha_comparison.png'))
        plt.close()
    
    # Plot Final Performance Bar Chart
    if 'eval_mean_reward' in dfs and 'eval_std_reward' in dfs:
        fig, ax = plt.subplots(dpi=args.dpi)
        
        # Get final evaluation rewards for each variant
        final_rewards = []
        
        for variant_name in dfs['eval_mean_reward']['variant_name'].unique():
            variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name]
            
            if args.max_steps:
                variant_data = variant_data[variant_data['step'] <= args.max_steps]
            
            # Get the last recorded reward and step
            last_idx = variant_data['step'].idxmax()
            last_step = variant_data.loc[last_idx, 'step']
            final_reward = variant_data.loc[last_idx, 'value']
            
            # Get the corresponding std value
            std_data = dfs['eval_std_reward']
            std_data = std_data[(std_data['variant_name'] == variant_name) & 
                               (std_data['step'] == last_step)]
            
            final_std = std_data['value'].values[0] if not std_data.empty else 0
            
            final_rewards.append({
                'variant_name': variant_name,
                'reward': final_reward,
                'std': final_std
            })
        
        # Convert to DataFrame and sort by reward
        final_df = pd.DataFrame(final_rewards)
        final_df = final_df.sort_values('reward', ascending=False)
        
        # Create bar plot
        bar_colors = []
        for variant in final_df['variant_name']:
            if 'Twin' in variant:
                if 'Adaptive' in variant:
                    bar_colors.append('blue')
                elif 'Fixed' in variant:
                    bar_colors.append('green')
                else:  # No Entropy
                    bar_colors.append('red')
            else:  # Single Critic
                if 'Adaptive' in variant:
                    bar_colors.append('purple')
                elif 'Fixed' in variant:
                    bar_colors.append('orange')
                else:  # No Entropy
                    bar_colors.append('brown')
        
        bars = ax.bar(
            final_df['variant_name'],
            final_df['reward'],
            yerr=final_df['std'],
            capsize=10,
            color=bar_colors,
            alpha=0.7
        )
        
        # Add value labels
        for bar, reward in zip(bars, final_df['reward']):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 5,
                f"{reward:.2f}",
                ha='center',
                va='bottom'
            )
        
        ax.set_xlabel('SAC Variant')
        ax.set_ylabel('Final Evaluation Reward')
        ax.set_title('SAC Variants: Final Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust figure for long variant names
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'final_performance.png'))
        plt.close()
    
    # Create comparison dashboard
    # This is a summary figure showing the most important metrics side by side
    fig = plt.figure(figsize=(15, 10), dpi=args.dpi)
    
    # 1. Evaluation rewards
    if 'eval_mean_reward' in dfs:
        ax1 = fig.add_subplot(221)
        
        for variant_name in dfs['eval_mean_reward']['variant_name'].unique():
            variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
            
            # Extract entropy type and critic type
            entropy_type = variant_name.split(' + ')[1]
            critic_type = variant_name.split(' + ')[0]
            
            # Sort by step
            variant_data = variant_data.sort_values('step')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                variant_data = variant_data[variant_data['step'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                # Group by step
                grouped = variant_data.groupby('step').agg({
                    'value': 'mean'
                }).reset_index()
                
                # Sort and apply smoothing
                grouped = grouped.sort_values('step')
                smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
            else:
                grouped = variant_data.sort_values('step')
                smoothed_rewards = grouped['value']
            
            # Plot with appropriate color and style
            color = colors.get(variant_name, 'gray')
            linestyle = line_styles.get(entropy_type, '-')
            
            ax1.plot(grouped['step'], smoothed_rewards, 
                    label=variant_name, linewidth=2,
                    color=color, linestyle=linestyle)
        
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Evaluation Reward')
        ax1.set_title('Evaluation Performance')
        ax1.legend(fontsize='small')
        ax1.grid(True, alpha=0.3)
    
    # 2. Training rewards
    if 'train_episode_reward' in dfs:
        ax2 = fig.add_subplot(222)
        
        for variant_name in dfs['train_episode_reward']['variant_name'].unique():
            variant_data = dfs['train_episode_reward'][dfs['train_episode_reward']['variant_name'] == variant_name].copy()
            
            # Extract entropy type
            entropy_type = variant_name.split(' + ')[1]
            
            # Bin by steps
            bin_size = 5000  # Adjust as needed
            variant_data['step_bin'] = (variant_data['step'] // bin_size) * bin_size
            
            # Group by step bin
            grouped = variant_data.groupby('step_bin').agg({
                'value': 'mean'
            }).reset_index()
            
            # Sort by step bin
            grouped = grouped.sort_values('step_bin')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                grouped = grouped[grouped['step_bin'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
            else:
                smoothed_rewards = grouped['value']
            
            # Plot with appropriate color and style
            color = colors.get(variant_name, 'gray')
            linestyle = line_styles.get(entropy_type, '-')
            
            ax2.plot(grouped['step_bin'], smoothed_rewards, 
                    label=variant_name, linewidth=2,
                    color=color, linestyle=linestyle)
        
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Mean Training Reward')
        ax2.set_title('Training Performance')
        ax2.legend(fontsize='small')
        ax2.grid(True, alpha=0.3)
    
    # 3. Policy Loss
    if 'train_policy_loss' in dfs:
        ax3 = fig.add_subplot(223)
        
        for variant_name in dfs['train_policy_loss']['variant_name'].unique():
            variant_data = dfs['train_policy_loss'][dfs['train_policy_loss']['variant_name'] == variant_name].copy()
            
            # Extract entropy type
            entropy_type = variant_name.split(' + ')[1]
            
            # Bin by steps
            bin_size = 5000  # Adjust as needed
            variant_data['step_bin'] = (variant_data['step'] // bin_size) * bin_size
            
            # Group by step bin
            grouped = variant_data.groupby('step_bin').agg({
                'value': 'mean'
            }).reset_index()
            
            # Sort by step bin
            grouped = grouped.sort_values('step_bin')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                grouped = grouped[grouped['step_bin'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_values = apply_smoothing(grouped['value'], args.smoothing)
            else:
                smoothed_values = grouped['value']
            
            # Plot with appropriate color and style
            color = colors.get(variant_name, 'gray')
            linestyle = line_styles.get(entropy_type, '-')
            
            ax3.plot(grouped['step_bin'], smoothed_values, 
                    label=variant_name, linewidth=2,
                    color=color, linestyle=linestyle)
        
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Policy Loss')
        ax3.set_title('Policy Loss Comparison')
        ax3.legend(fontsize='small')
        ax3.grid(True, alpha=0.3)
    
    # 4. Alpha or Critic Loss
    if 'train_alpha' in dfs:
        ax4 = fig.add_subplot(224)
        
        adaptive_variants = [v for v in dfs['train_alpha']['variant_name'].unique() 
                           if 'Adaptive' in v]
        
        for variant_name in adaptive_variants:
            variant_data = dfs['train_alpha'][dfs['train_alpha']['variant_name'] == variant_name].copy()
            
            # Bin by steps
            bin_size = 5000  # Adjust as needed
            variant_data['step_bin'] = (variant_data['step'] // bin_size) * bin_size
            
            # Group by step bin
            grouped = variant_data.groupby('step_bin').agg({
                'value': 'mean'
            }).reset_index()
            
            # Sort by step bin
            grouped = grouped.sort_values('step_bin')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                grouped = grouped[grouped['step_bin'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_values = apply_smoothing(grouped['value'], args.smoothing)
            else:
                smoothed_values = grouped['value']
            
            # Plot with appropriate color
            color = colors.get(variant_name, 'gray')
            
            ax4.plot(grouped['step_bin'], smoothed_values, 
                    label=variant_name, linewidth=2, color=color)
        
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Alpha Value')
        ax4.set_title('Temperature Parameter (Alpha) - Adaptive Variants')
        ax4.legend(fontsize='small')
        ax4.grid(True, alpha=0.3)
    elif 'train_critic_1_loss' in dfs:
        # Use critic loss if alpha isn't available
        ax4 = fig.add_subplot(224)
        
        for variant_name in dfs['train_critic_1_loss']['variant_name'].unique():
            variant_data = dfs['train_critic_1_loss'][dfs['train_critic_1_loss']['variant_name'] == variant_name].copy()
            
            # Extract entropy type
            entropy_type = variant_name.split(' + ')[1]
            
            # Bin by steps
            bin_size = 5000  # Adjust as needed
            variant_data['step_bin'] = (variant_data['step'] // bin_size) * bin_size
            
            # Group by step bin
            grouped = variant_data.groupby('step_bin').agg({
                'value': 'mean'
            }).reset_index()
            
            # Sort by step bin
            grouped = grouped.sort_values('step_bin')
            
            # Limit steps if max_steps is specified
            if args.max_steps:
                grouped = grouped[grouped['step_bin'] <= args.max_steps]
            
            # Apply smoothing
            if args.smoothing > 1:
                smoothed_values = apply_smoothing(grouped['value'], args.smoothing)
            else:
                smoothed_values = grouped['value']
            
            # Plot with appropriate color and style
            color = colors.get(variant_name, 'gray')
            linestyle = line_styles.get(entropy_type, '-')
            
            ax4.plot(grouped['step_bin'], smoothed_values, 
                    label=variant_name, linewidth=2,
                    color=color, linestyle=linestyle)
        
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Critic 1 Loss')
        ax4.set_title('Critic Loss Comparison')
        ax4.legend(fontsize='small')
        ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('SAC Variants Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'sac_variants_dashboard.png'))
    plt.close()
    
    # Create Critic Type Comparison Dashboard
    fig = plt.figure(figsize=(15, 10), dpi=args.dpi)
    
    # 1. Twin vs Single - Adaptive Entropy
    if 'eval_mean_reward' in dfs:
        ax1 = fig.add_subplot(221)
        
        variants = [
            'Twin Critic + Adaptive Entropy',
            'Single Critic + Adaptive Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax1.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Evaluation Reward')
        ax1.set_title('Twin vs Single Critic (Adaptive Entropy)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Twin vs Single - Fixed Entropy
    if 'eval_mean_reward' in dfs:
        ax2 = fig.add_subplot(222)
        
        variants = [
            'Twin Critic + Fixed Entropy',
            'Single Critic + Fixed Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax2.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Mean Evaluation Reward')
        ax2.set_title('Twin vs Single Critic (Fixed Entropy)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Twin vs Single - No Entropy
    if 'eval_mean_reward' in dfs:
        ax3 = fig.add_subplot(223)
        
        variants = [
            'Twin Critic + No Entropy',
            'Single Critic + No Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax3.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Mean Evaluation Reward')
        ax3.set_title('Twin vs Single Critic (No Entropy)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Twin Critic - All Entropy Types
    if 'eval_mean_reward' in dfs:
        ax4 = fig.add_subplot(224)
        
        variants = [
            'Twin Critic + Adaptive Entropy',
            'Twin Critic + Fixed Entropy',
            'Twin Critic + No Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax4.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Mean Evaluation Reward')
        ax4.set_title('Twin Critic: Entropy Types Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Critic Type and Entropy Regularization Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'critic_entropy_comparison.png'))
    plt.close()
    
    # Create Entropy Type Comparison Dashboard
    fig = plt.figure(figsize=(15, 10), dpi=args.dpi)
    
    # 1. Adaptive vs Fixed vs None - Twin Critic
    if 'eval_mean_reward' in dfs:
        ax1 = fig.add_subplot(221)
        
        variants = [
            'Twin Critic + Adaptive Entropy',
            'Twin Critic + Fixed Entropy',
            'Twin Critic + No Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax1.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Evaluation Reward')
        ax1.set_title('Twin Critic: Entropy Types Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Adaptive vs Fixed vs None - Single Critic
    if 'eval_mean_reward' in dfs:
        ax2 = fig.add_subplot(222)
        
        variants = [
            'Single Critic + Adaptive Entropy',
            'Single Critic + Fixed Entropy',
            'Single Critic + No Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax2.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Mean Evaluation Reward')
        ax2.set_title('Single Critic: Entropy Types Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Adaptive Entropy - Twin vs Single
    if 'eval_mean_reward' in dfs:
        ax3 = fig.add_subplot(223)
        
        variants = [
            'Twin Critic + Adaptive Entropy',
            'Single Critic + Adaptive Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax3.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Mean Evaluation Reward')
        ax3.set_title('Adaptive Entropy: Twin vs Single Critic')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. No Entropy - Twin vs Single
    if 'eval_mean_reward' in dfs:
        ax4 = fig.add_subplot(224)
        
        variants = [
            'Twin Critic + No Entropy',
            'Single Critic + No Entropy'
        ]
        
        for variant_name in variants:
            if variant_name in dfs['eval_mean_reward']['variant_name'].unique():
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant_name].copy()
                
                # Sort by step
                variant_data = variant_data.sort_values('step')
                
                # Limit steps if max_steps is specified
                if args.max_steps:
                    variant_data = variant_data[variant_data['step'] <= args.max_steps]
                
                # Apply smoothing
                if args.smoothing > 1:
                    # Group by step
                    grouped = variant_data.groupby('step').agg({
                        'value': 'mean'
                    }).reset_index()
                    
                    # Sort and apply smoothing
                    grouped = grouped.sort_values('step')
                    smoothed_rewards = apply_smoothing(grouped['value'], args.smoothing)
                else:
                    grouped = variant_data.sort_values('step')
                    smoothed_rewards = grouped['value']
                
                # Plot
                color = colors.get(variant_name, 'gray')
                ax4.plot(grouped['step'], smoothed_rewards, 
                        label=variant_name, linewidth=2, color=color)
        
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Mean Evaluation Reward')
        ax4.set_title('No Entropy: Twin vs Single Critic')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle('Entropy Type Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'entropy_type_comparison.png'))
    plt.close()

def generate_statistical_analysis(dfs, output_dir):
    """Generate statistical analysis of the results."""
    
    # Create a text file to store the analysis
    stats_file = os.path.join(output_dir, 'statistical_analysis.txt')
    
    with open(stats_file, 'w') as f:
        f.write("# Statistical Analysis of SAC Variants Performance\n\n")
        
        if 'eval_mean_reward' in dfs:
            f.write("## Evaluation Rewards Analysis\n\n")
            
            # Group evaluation rewards by variant
            variants = dfs['eval_mean_reward']['variant_name'].unique()
            
            rewards_by_variant = {}
            for variant in variants:
                variant_data = dfs['eval_mean_reward'][dfs['eval_mean_reward']['variant_name'] == variant]
                rewards_by_variant[variant] = variant_data['value'].values
            
            # Basic statistics for each variant
            f.write("### Basic Statistics\n\n")
            f.write("| Variant | Mean | Median | Std Dev | Min | Max | Final |\n")
            f.write("|---------|------|--------|---------|-----|-----|-------|\n")
            
            for variant, rewards in rewards_by_variant.items():
                last_reward = rewards[-1]
                f.write(f"| {variant} | {np.mean(rewards):.2f} | {np.median(rewards):.2f} | "
                       f"{np.std(rewards):.2f} | {np.min(rewards):.2f} | {np.max(rewards):.2f} | "
                       f"{last_reward:.2f} |\n")
            
            f.write("\n")
            
            # Comparative statistics if we have multiple variants
            if len(variants) > 1:
                f.write("### Comparative Analysis\n\n")
                
                # Find the best, worst, and most stable variants
                mean_rewards = {variant: np.mean(rewards) for variant, rewards in rewards_by_variant.items()}
                best_variant = max(mean_rewards, key=mean_rewards.get)
                worst_variant = min(mean_rewards, key=mean_rewards.get)
                
                cv_values = {variant: np.std(rewards) / abs(np.mean(rewards)) 
                            for variant, rewards in rewards_by_variant.items()}
                most_stable = min(cv_values, key=cv_values.get)
                least_stable = max(cv_values, key=cv_values.get)
                
                f.write(f"**Best variant (by mean reward):** {best_variant} with mean reward of {mean_rewards[best_variant]:.2f}\n\n")
                f.write(f"**Worst variant (by mean reward):** {worst_variant} with mean reward of {mean_rewards[worst_variant]:.2f}\n\n")
                f.write(f"**Most stable variant (lowest coefficient of variation):** {most_stable} with CV of {cv_values[most_stable]:.4f}\n\n")
                f.write(f"**Least stable variant (highest coefficient of variation):** {least_stable} with CV of {cv_values[least_stable]:.4f}\n\n")
                
                f.write("### Pairwise Statistical Comparisons\n\n")
                
                # Organize variants for pairwise comparisons
                variant_pairs = []
                for i in range(len(variants)):
                    for j in range(i+1, len(variants)):
                        variant_pairs.append((variants[i], variants[j]))
                
                for variant1, variant2 in variant_pairs:
                    f.write(f"**{variant1} vs {variant2}**\n\n")
                    
                    # T-test for comparing means
                    t_stat, p_value = stats.ttest_ind(
                        rewards_by_variant[variant1], 
                        rewards_by_variant[variant2],
                        equal_var=False  # Welch's t-test (doesn't assume equal variance)
                    )
                    
                    f.write(f"- t-test: t={t_stat:.3f}, p={p_value:.5f}")
                    if p_value < 0.05:
                        f.write(" (statistically significant difference)\n")
                    else:
                        f.write(" (no statistically significant difference)\n")
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, p_value = stats.mannwhitneyu(
                        rewards_by_variant[variant1], 
                        rewards_by_variant[variant2]
                    )
                    
                    f.write(f"- Mann-Whitney U test: U={u_stat:.3f}, p={p_value:.5f}")
                    if p_value < 0.05:
                        f.write(" (statistically significant difference)\n")
                    else:
                        f.write(" (no statistically significant difference)\n")
                    
                    # Effect size (Cohen's d)
                    mean1, mean2 = np.mean(rewards_by_variant[variant1]), np.mean(rewards_by_variant[variant2])
                    std1, std2 = np.std(rewards_by_variant[variant1]), np.std(rewards_by_variant[variant2])
                    
                    # Pooled standard deviation
                    n1, n2 = len(rewards_by_variant[variant1]), len(rewards_by_variant[variant2])
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
                
                # Final performance comparison
                f.write("### Final Performance Comparison\n\n")
                
                final_rewards = {}
                for variant, rewards in rewards_by_variant.items():
                    final_rewards[variant] = rewards[-1]
                
                best_final = max(final_rewards.values())
                
                f.write("| Variant | Final Reward | % of Best |\n")
                f.write("|---------|-------------|----------|\n")
                
                for variant, reward in sorted(final_rewards.items(), key=lambda x: x[1], reverse=True):
                    pct_of_best = (reward / best_final) * 100
                    f.write(f"| {variant} | {reward:.2f} | {pct_of_best:.1f}% |\n")
                
                f.write("\n")
            
            # Analysis by critic type
            f.write("## Analysis by Critic Type\n\n")
            
            twin_critic_variants = [v for v in variants if 'Twin Critic' in v]
            single_critic_variants = [v for v in variants if 'Single Critic' in v]
            
            # Compare average performance by critic type
            if twin_critic_variants and single_critic_variants:
                twin_rewards = np.concatenate([rewards_by_variant[v] for v in twin_critic_variants])
                single_rewards = np.concatenate([rewards_by_variant[v] for v in single_critic_variants])
                
                twin_mean = np.mean(twin_rewards)
                single_mean = np.mean(single_rewards)
                
                f.write("### Twin Critic vs Single Critic\n\n")
                f.write(f"- **Twin Critic Average Reward:** {twin_mean:.2f}\n")
                f.write(f"- **Single Critic Average Reward:** {single_mean:.2f}\n\n")
                
                # T-test between twin and single critic
                t_stat, p_value = stats.ttest_ind(
                    twin_rewards, 
                    single_rewards,
                    equal_var=False
                )
                
                f.write(f"- t-test: t={t_stat:.3f}, p={p_value:.5f}")
                if p_value < 0.05:
                    f.write(" (statistically significant difference)\n")
                else:
                    f.write(" (no statistically significant difference)\n")
                
                # Effect size
                twin_std = np.std(twin_rewards)
                single_std = np.std(single_rewards)
                pooled_std = np.sqrt(((len(twin_rewards)-1)*twin_std**2 + 
                                     (len(single_rewards)-1)*single_std**2) / 
                                    (len(twin_rewards) + len(single_rewards) - 2))
                
                cohen_d = abs(twin_mean - single_mean) / pooled_std
                
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
                
                # Compare stability
                twin_cv = np.std(twin_rewards) / abs(np.mean(twin_rewards))
                single_cv = np.std(single_rewards) / abs(np.mean(single_rewards))
                
                f.write(f"- **Twin Critic Stability (CV):** {twin_cv:.4f}\n")
                f.write(f"- **Single Critic Stability (CV):** {single_cv:.4f}\n\n")
                
                if twin_cv < single_cv:
                    f.write("Twin Critic variants showed **more stable** performance than Single Critic variants.\n\n")
                else:
                    f.write("Single Critic variants showed **more stable** performance than Twin Critic variants.\n\n")
            
            # Analysis by entropy type
            f.write("## Analysis by Entropy Type\n\n")
            
            adaptive_variants = [v for v in variants if 'Adaptive Entropy' in v]
            fixed_variants = [v for v in variants if 'Fixed Entropy' in v]
            no_entropy_variants = [v for v in variants if 'No Entropy' in v]
            
            entropy_types = []
            if adaptive_variants:
                entropy_types.append(('Adaptive Entropy', 
                                    np.concatenate([rewards_by_variant[v] for v in adaptive_variants])))
            if fixed_variants:
                entropy_types.append(('Fixed Entropy', 
                                    np.concatenate([rewards_by_variant[v] for v in fixed_variants])))
            if no_entropy_variants:
                entropy_types.append(('No Entropy', 
                                    np.concatenate([rewards_by_variant[v] for v in no_entropy_variants])))
            
            if len(entropy_types) > 1:
                f.write("### Comparison by Entropy Type\n\n")
                f.write("| Entropy Type | Mean Reward | Std Dev | CV |\n")
                f.write("|-------------|------------|---------|----|\n")
                
                for entropy_type, rewards in entropy_types:
                    mean_reward = np.mean(rewards)
                    std_dev = np.std(rewards)
                    cv = std_dev / abs(mean_reward)
                    
                    f.write(f"| {entropy_type} | {mean_reward:.2f} | {std_dev:.2f} | {cv:.4f} |\n")
                
                f.write("\n")
                
                # Find best entropy type
                means = {ent_type: np.mean(rewards) for ent_type, rewards in entropy_types}
                best_entropy = max(means, key=means.get)
                
                cvs = {ent_type: np.std(rewards) / abs(np.mean(rewards)) for ent_type, rewards in entropy_types}
                most_stable_entropy = min(cvs, key=cvs.get)
                
                f.write(f"**Best performing entropy type:** {best_entropy} with mean reward of {means[best_entropy]:.2f}\n\n")
                f.write(f"**Most stable entropy type:** {most_stable_entropy} with CV of {cvs[most_stable_entropy]:.4f}\n\n")
                
                # Pairwise comparisons between entropy types
                f.write("### Pairwise Comparisons Between Entropy Types\n\n")
                
                entropy_pairs = []
                for i in range(len(entropy_types)):
                    for j in range(i+1, len(entropy_types)):
                        entropy_pairs.append((entropy_types[i][0], entropy_types[j][0],
                                             entropy_types[i][1], entropy_types[j][1]))
                
                for ent_type1, ent_type2, rewards1, rewards2 in entropy_pairs:
                    f.write(f"**{ent_type1} vs {ent_type2}**\n\n")
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(
                        rewards1, 
                        rewards2,
                        equal_var=False
                    )
                    
                    f.write(f"- t-test: t={t_stat:.3f}, p={p_value:.5f}")
                    if p_value < 0.05:
                        f.write(" (statistically significant difference)\n")
                    else:
                        f.write(" (no statistically significant difference)\n")
                    
                    # Effect size
                    mean1, mean2 = np.mean(rewards1), np.mean(rewards2)
                    std1, std2 = np.std(rewards1), np.std(rewards2)
                    n1, n2 = len(rewards1), len(rewards2)
                    
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
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
            
            # Comprehensive analysis of each variant
            f.write("## Detailed Analysis of Each Variant\n\n")
            
            for variant in variants:
                f.write(f"### {variant}\n\n")
                
                rewards = rewards_by_variant[variant]
                mean_reward = np.mean(rewards)
                final_reward = rewards[-1]
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                std_dev = np.std(rewards)
                cv = std_dev / abs(mean_reward)
                
                f.write(f"- **Mean Reward:** {mean_reward:.2f}\n")
                f.write(f"- **Final Reward:** {final_reward:.2f}\n")
                f.write(f"- **Min Reward:** {min_reward:.2f}\n")
                f.write(f"- **Max Reward:** {max_reward:.2f}\n")
                f.write(f"- **Standard Deviation:** {std_dev:.2f}\n")
                f.write(f"- **Coefficient of Variation:** {cv:.4f}\n\n")
                
                f.write(f"**Performance Summary:** This variant achieved {(final_reward / best_final * 100):.1f}% ")
                f.write(f"of the best final performance and {(mean_reward / means[best_entropy] * 100):.1f}% ")
                f.write(f"of the best mean performance.\n\n")
            
            # Overall conclusions
            f.write("## Overall Conclusions\n\n")
            
            # Rank variants by final performance
            ranked_variants = sorted(final_rewards.items(), key=lambda x: x[1], reverse=True)
            
            f.write("### Rankings by Final Performance\n\n")
            f.write("| Rank | Variant | Final Reward |\n")
            f.write("|------|---------|-------------|\n")
            
            for rank, (variant, reward) in enumerate(ranked_variants, 1):
                f.write(f"| {rank} | {variant} | {reward:.2f} |\n")
            
            f.write("\n")
            
            # Summarize the key findings
            f.write("### Key Findings\n\n")
            
            # Critic type comparison summary
            if twin_critic_variants and single_critic_variants:
                if twin_mean > single_mean:
                    f.write("1. **Twin Critic** variants generally outperformed **Single Critic** variants ")
                    f.write(f"(average reward {twin_mean:.2f} vs {single_mean:.2f}, ")
                    f.write(f"a {(twin_mean / single_mean * 100 - 100):.1f}% improvement).\n\n")
                else:
                    f.write("1. **Single Critic** variants generally outperformed **Twin Critic** variants ")
                    f.write(f"(average reward {single_mean:.2f} vs {twin_mean:.2f}, ")
                    f.write(f"a {(single_mean / twin_mean * 100 - 100):.1f}% improvement).\n\n")
            
            # Entropy type comparison summary
            if len(entropy_types) > 1:
                f.write(f"2. **{best_entropy}** performed best among entropy types ")
                f.write(f"with a mean reward of {means[best_entropy]:.2f}.\n\n")
                
                f.write(f"3. **{most_stable_entropy}** showed the most stable training ")
                f.write(f"with a coefficient of variation of {cvs[most_stable_entropy]:.4f}.\n\n")
            
            # Best overall variant
            f.write(f"4. The best overall variant was **{ranked_variants[0][0]}** ")
            f.write(f"with a final reward of {ranked_variants[0][1]:.2f}.\n\n")
            
            # Worst overall variant
            f.write(f"5. The worst performing variant was **{ranked_variants[-1][0]}** ")
            f.write(f"with a final reward of {ranked_variants[-1][1]:.2f}.\n\n")
            
            # General recommendations based on findings
            f.write("### Recommendations\n\n")
            
            if twin_mean > single_mean:
                f.write("1. Use **Twin Critic** implementation for better performance.\n")
            else:
                f.write("1. Consider using **Single Critic** implementation for better performance.\n")
            
            f.write(f"2. **{best_entropy}** is recommended for this specific environment.\n")
            
            if 'Twin Critic + Adaptive Entropy' in variants or 'Single Critic + Adaptive Entropy' in variants:
                if 'Adaptive Entropy' == best_entropy:
                    f.write("3. The automatic entropy tuning mechanism is beneficial and should be enabled.\n")
                else:
                    f.write("3. Consider disabling automatic entropy tuning in favor of ")
                    f.write(f"a {best_entropy.lower()}.\n")
            
            f.write("4. For future work, consider further hyperparameter tuning for the best performing variant ")
            f.write(f"(**{ranked_variants[0][0]}**) to potentially achieve even better results.\n")

def main():
    """Main function for analyzing results."""
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