import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def calculate_metrics(df, label):
    # Exclude first 10 seconds (transient)
    df = df[df['timestamp'] > 10.0].copy()
    
    # UGV Tracking Error
    errors = np.sqrt((df['x'] - df['target_x'])**2 + (df['y'] - df['target_y'])**2)
    mean_err = errors.mean()
    max_err = errors.max()
    
    # Control Smoothness
    dv = np.diff(df['v_cmd'])
    dw = np.diff(df['w_cmd'])
    smoothness = np.mean(np.sqrt(dv**2 + dw**2))
    
    # Uncertainty
    mean_sigma = df['cov_trace'].mean()
    max_sigma = df['cov_trace'].max()
    
    # Visibility
    visibility_pct = (df['aruco_visible'] == 1).mean() * 100.0
    
    # Max Dropout (s)
    visible = df['aruco_visible'].values
    max_drop = 0
    curr_drop = 0
    dt_avg = np.mean(np.diff(df['timestamp']))
    for v in visible:
        if v == 0:
            curr_drop += dt_avg
        else:
            max_drop = max(max_drop, curr_drop)
            curr_drop = 0
    max_drop = max(max_drop, curr_drop)
    
    # View Quality
    mean_qk = df['q_k'].mean()
    min_qk = df['q_k'].min()
    
    return {
        'Condition': label,
        'Mean Err (m)': mean_err,
        'Max Err (m)': max_err,
        'Smoothness': smoothness,
        'Mean tr(Sigma)': mean_sigma,
        'Max tr(Sigma)': max_sigma,
        'Visibility %': visibility_pct,
        'Max Dropout (s)': max_drop,
        'Mean q_k': mean_qk,
        'Min q_k': min_qk
    }

def main():
    log_dir = os.path.expanduser('~/experiment_logs')
    results_dir = os.path.expanduser('~/Desktop/Project_ws/results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Find files (newest first for each pattern)
    c_files = sorted(glob.glob(os.path.join(log_dir, 'centroid_fallback_*.csv')), reverse=True)
    m_files = sorted(glob.glob(os.path.join(log_dir, 'full_mpc_20*.csv')), reverse=True) # Avoid catching 'hardened'
    h_files = sorted(glob.glob(os.path.join(log_dir, 'full_mpc_hardened_*.csv')), reverse=True)
    
    results = []
    
    if c_files:
        df_c = pd.read_csv(c_files[0])
        results.append(calculate_metrics(df_c, "Centroid Baseline"))
        
    if m_files:
        # Filter out hardened if glob captured them
        non_hardened = [f for f in m_files if 'hardened' not in f]
        if non_hardened:
            df_m = pd.read_csv(non_hardened[0])
            results.append(calculate_metrics(df_m, "Original MPC"))

    if h_files:
        df_h = pd.read_csv(h_files[0])
        results.append(calculate_metrics(df_h, "Hardened MPC"))
    
    table_df = pd.DataFrame(results)
    table_df.to_csv(os.path.join(results_dir, 'final_comparison.csv'), index=False)
    
    print("\nFINAL COMPARATIVE EVALUATION")
    print(table_df.to_string())
    
    # Plots for the three conditions
    colors = {'Centroid Baseline': 'blue', 'Original MPC': 'red', 'Hardened MPC': 'green'}
    
    plt.figure(figsize=(10, 6))
    for res in results:
        label = res['Condition']
        if label == "Centroid Baseline": df = pd.read_csv(c_files[0])
        elif label == "Original MPC": df = pd.read_csv(non_hardened[0])
        elif label == "Hardened MPC": df = pd.read_csv(h_files[0])
        
        dfp = df[df['timestamp'] > 10.0]
        qk = dfp['q_k']
        plt.plot(dfp['timestamp'], qk, label=label, color=colors[label], alpha=0.7)
    
    plt.axhline(0.1, color='black', linestyle='--', label='Visibility Threshold')
    plt.title('View Quality ($q_k$) Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('$q_k$')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'final_view_quality.png'))
    
    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    main()
