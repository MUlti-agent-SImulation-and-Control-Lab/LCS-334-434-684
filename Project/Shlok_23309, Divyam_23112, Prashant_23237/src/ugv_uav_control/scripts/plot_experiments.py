#!/usr/bin/env python3
"""
Controller Comparison Plotter
Generates comparison plots from experiment CSV files.

Usage:
    python3 plot_experiments.py <csv_file1> [<csv_file2> ...]

Labels are extracted from experiment_name parameter (first part before timestamp).
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re


def load_experiment(filepath):
    """Load CSV and return dataframe."""
    df = pd.read_csv(filepath)
    return df


def extract_label(filepath):
    """
    Extract a readable label from filename.
    Filename format: <exp_name>_YYYYMMDD_HHMMSS.csv
    Returns <exp_name> with underscores replaced by spaces and title-cased.
    """
    basename = os.path.basename(filepath).replace('.csv', '')
    # Remove trailing timestamp: _YYYYMMDD_HHMMSS
    label = re.sub(r'_\d{8}_\d{6}$', '', basename)
    return label.replace('_', ' ').title()


def compute_metrics(df):
    """Compute summary metrics from a dataframe."""
    tracking_err = np.sqrt(
        (df['x'] - df['target_x'])**2 + (df['y'] - df['target_y'])**2)

    dv = np.diff(df['v_cmd'].values)
    dw = np.diff(df['w_cmd'].values)
    control_jerk = np.sqrt(dv**2 + dw**2)

    return {
        'mean_tracking_error': tracking_err.mean(),
        'max_tracking_error': tracking_err.max(),
        'control_smoothness': control_jerk.mean() if len(control_jerk) > 0 else 0,
        'mean_cov_trace': df['cov_trace'].mean(),
        'max_cov_trace': df['cov_trace'].max(),
        'fov_exit_pct': (df['fov_distance'] > 5.77).mean() * 100,
    }


def plot_comparison(exp_files, labels):
    """Generate 6-panel time-series comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Controller Comparison', fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (filepath, label) in enumerate(zip(exp_files, labels)):
        df = load_experiment(filepath)
        c = colors[i % len(colors)]

        # 1. Trajectory (x vs y)
        axes[0, 0].plot(df['x'], df['y'], label=label, color=c,
                        alpha=0.8, linewidth=1.5)

        # 2. Tracking Error vs Time
        error = np.sqrt(
            (df['x'] - df['target_x'])**2 + (df['y'] - df['target_y'])**2)
        axes[0, 1].plot(df['timestamp'], error, label=label, color=c,
                        alpha=0.7, linewidth=1)

        # 3. Covariance Trace vs Time
        axes[0, 2].plot(df['timestamp'], df['cov_trace'], label=label,
                        color=c, alpha=0.7, linewidth=1)

        # 4. Linear Velocity vs Time
        axes[1, 0].plot(df['timestamp'], df['v_cmd'],
                        label=label, color=c, alpha=0.7, linewidth=1)

        # 5. Angular Velocity vs Time
        axes[1, 1].plot(df['timestamp'], df['w_cmd'],
                        label=label, color=c, alpha=0.7, linewidth=1)

        # 6. FoV Distance vs Time
        axes[1, 2].plot(df['timestamp'], df['fov_distance'],
                        label=label, color=c, alpha=0.7, linewidth=1)

    # --- Formatting ---
    # Trajectory
    axes[0, 0].set_title('Trajectory')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    circle = plt.Circle((0, 0), 5.77, fill=False, color='red',
                         linestyle='--', alpha=0.5, label='FoV')
    axes[0, 0].add_patch(circle)
    axes[0, 0].set_aspect('equal', adjustable='datalim')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Tracking Error
    axes[0, 1].set_title('Tracking Error')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Error (m)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Covariance
    axes[0, 2].set_title('Covariance Trace')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('tr(Σ)')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Linear Velocity
    axes[1, 0].set_title('Linear Velocity')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('v (m/s)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Angular Velocity
    axes[1, 1].set_title('Angular Velocity')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('ω (rad/s)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # FoV Distance
    axes[1, 2].set_title('Distance from FoV Center')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('d (m)')
    axes[1, 2].axhline(y=5.77, color='red', linestyle='--',
                        alpha=0.5, label='FoV boundary')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = os.path.expanduser('~/experiment_logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')
    plt.show()


def plot_bar_summary(exp_files, labels):
    """Generate bar chart comparing summary metrics across controllers."""
    all_metrics = [compute_metrics(load_experiment(f)) for f in exp_files]

    metric_keys = [
        'mean_tracking_error', 'control_smoothness',
        'mean_cov_trace', 'fov_exit_pct']
    titles = [
        'Mean Tracking Error (m)', 'Control Smoothness (jerk)',
        'Mean Covariance tr(Σ)', 'FoV Exit Rate (%)']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Controller Summary Comparison', fontsize=14,
                 fontweight='bold')

    for i, (key, title) in enumerate(zip(metric_keys, titles)):
        values = [m[key] for m in all_metrics]
        bar_colors = [colors[j % len(colors)] for j in range(len(labels))]
        bars = axes[i].bar(labels, values, color=bar_colors)
        axes[i].set_title(title, fontsize=10)
        axes[i].set_ylabel(key.replace('_', ' '), fontsize=8)

        # Value labels on bars
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    out_dir = os.path.expanduser('~/experiment_logs')
    out_path = os.path.join(out_dir, 'bar_comparison.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')
    plt.show()


def print_summary_table(exp_files, labels):
    """Print a text summary table."""
    print('\n' + '=' * 80)
    print(f'{"Metric":<25}', end='')
    for label in labels:
        print(f'{label:>15}', end='')
    print()
    print('-' * 80)

    all_metrics = [compute_metrics(load_experiment(f)) for f in exp_files]

    for key in all_metrics[0].keys():
        print(f'{key:<25}', end='')
        for m in all_metrics:
            print(f'{m[key]:>15.4f}', end='')
        print()
    print('=' * 80 + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 plot_experiments.py <csv1> [csv2] ...')
        print()
        print('Example:')
        print('  python3 plot_experiments.py \\')
        print('      ~/experiment_logs/mpc_*.csv \\')
        print('      ~/experiment_logs/belief_mpc_*.csv')
        sys.exit(1)

    files = sys.argv[1:]
    labels = [extract_label(f) for f in files]

    print(f'Comparing {len(files)} experiments: {labels}')

    # Print text table
    print_summary_table(files, labels)

    # Generate plots
    plot_comparison(files, labels)
    plot_bar_summary(files, labels)