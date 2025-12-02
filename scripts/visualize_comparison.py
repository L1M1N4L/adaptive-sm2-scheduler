"""
Create research-quality visualizations comparing schedulers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style for research papers
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def load_comparison_data(results_path: str, metrics_path: str):
    """Load comparison results."""
    results_df = pd.read_csv(results_path)
    metrics_df = pd.read_csv(metrics_path)
    return results_df, metrics_df

def plot_prediction_accuracy(results_df: pd.DataFrame, output_dir: Path):
    """Plot prediction accuracy comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    schedulers = results_df['scheduler'].unique()
    colors = sns.color_palette("husl", len(schedulers))
    
    # 1. MAE by scheduler
    ax = axes[0, 0]
    mae_data = []
    for sched in schedulers:
        subset = results_df[results_df['scheduler'] == sched]
        errors = np.abs(subset['p_recall_pred'] - subset['p_recall_actual'])
        mae_data.append({
            'scheduler': sched,
            'mae': errors.mean(),
            'std': errors.std()
        })
    mae_df = pd.DataFrame(mae_data)
    ax.bar(mae_df['scheduler'], mae_df['mae'], yerr=mae_df['std'], capsize=5, color=colors)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Accuracy (MAE)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Prediction vs Actual scatter
    ax = axes[0, 1]
    for i, sched in enumerate(schedulers):
        subset = results_df[results_df['scheduler'] == sched]
        ax.scatter(subset['p_recall_actual'], subset['p_recall_pred'], 
                  alpha=0.5, label=sched, color=colors[i], s=20)
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Recall Probability')
    ax.set_ylabel('Predicted Recall Probability')
    ax.set_title('Prediction vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax = axes[1, 0]
    for i, sched in enumerate(schedulers):
        subset = results_df[results_df['scheduler'] == sched]
        errors = subset['p_recall_pred'] - subset['p_recall_actual']
        ax.hist(errors, bins=30, alpha=0.6, label=sched, color=colors[i], density=True)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # 4. Interval comparison
    ax = axes[1, 1]
    interval_data = []
    for sched in schedulers:
        subset = results_df[results_df['scheduler'] == sched]
        interval_data.append({
            'scheduler': sched,
            'mean': subset['interval'].mean(),
            'median': subset['interval'].median(),
            'std': subset['interval'].std()
        })
    interval_df = pd.DataFrame(interval_data)
    x = np.arange(len(interval_df))
    width = 0.35
    ax.bar(x - width/2, interval_df['mean'], width, label='Mean', yerr=interval_df['std'], capsize=5)
    ax.bar(x + width/2, interval_df['median'], width, label='Median')
    ax.set_ylabel('Interval (days)')
    ax.set_title('Review Interval Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(interval_df['scheduler'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'prediction_accuracy.png'}")

def plot_ml_contribution(results_df: pd.DataFrame, output_dir: Path):
    """Plot ML contribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter for hybrid schedulers
    hybrid_data = results_df[results_df['scheduler'].str.contains('Hybrid', case=False)]
    
    if len(hybrid_data) > 0:
        # 1. ML confidence distribution
        ax = axes[0, 0]
        for sched in hybrid_data['scheduler'].unique():
            subset = hybrid_data[hybrid_data['scheduler'] == sched]
            ax.hist(subset['ml_confidence'], bins=30, alpha=0.6, label=sched, density=True)
        ax.set_xlabel('ML Confidence')
        ax.set_ylabel('Density')
        ax.set_title('ML Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. ML confidence vs review number
        ax = axes[0, 1]
        for sched in hybrid_data['scheduler'].unique():
            subset = hybrid_data[hybrid_data['scheduler'] == sched]
            grouped = subset.groupby('review_num')['ml_confidence'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=sched, linewidth=2)
        ax.set_xlabel('Review Number')
        ax.set_ylabel('Average ML Confidence')
        ax.set_title('ML Contribution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Interval difference (Hybrid vs SM-2)
        ax = axes[1, 0]
        sm2_intervals = results_df[results_df['scheduler'] == 'SM-2'].set_index(['item_id', 'review_num'])['interval']
        for sched in hybrid_data['scheduler'].unique():
            subset = hybrid_data[hybrid_data['scheduler'] == sched]
            subset_indexed = subset.set_index(['item_id', 'review_num'])
            diff = subset_indexed['interval'] - sm2_intervals
            diff = diff.dropna()
            if len(diff) > 0:
                ax.hist(diff, bins=30, alpha=0.6, label=sched, density=True)
        ax.set_xlabel('Interval Difference (Hybrid - SM-2)')
        ax.set_ylabel('Density')
        ax.set_title('Interval Adjustment by ML')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
        # 4. ML confidence vs prediction error
        ax = axes[1, 1]
        for sched in hybrid_data['scheduler'].unique():
            subset = hybrid_data[hybrid_data['scheduler'] == sched]
            errors = np.abs(subset['p_recall_pred'] - subset['p_recall_actual'])
            ax.scatter(subset['ml_confidence'], errors, alpha=0.5, label=sched, s=20)
        ax.set_xlabel('ML Confidence')
        ax.set_ylabel('Prediction Error (MAE)')
        ax.set_title('ML Confidence vs Prediction Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'ml_contribution.png'}")

def plot_metrics_comparison(metrics_df: pd.DataFrame, output_dir: Path):
    """Plot metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    schedulers = metrics_df['scheduler'].values
    x = np.arange(len(schedulers))
    width = 0.6
    
    # 1. MAE comparison
    ax = axes[0, 0]
    ax.bar(x, metrics_df['mae_p'], width, capsize=5)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Accuracy (MAE)')
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Average interval
    ax = axes[0, 1]
    ax.bar(x, metrics_df['avg_interval'], width, capsize=5)
    ax.set_ylabel('Average Interval (days)')
    ax.set_title('Review Interval')
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Average half-life
    ax = axes[1, 0]
    ax.bar(x, metrics_df['avg_halflife'], width, capsize=5)
    ax.set_ylabel('Average Half-life (days)')
    ax.set_title('Memory Half-life')
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. ML confidence (for hybrid schedulers)
    ax = axes[1, 1]
    hybrid_metrics = metrics_df[metrics_df['scheduler'].str.contains('Hybrid', case=False)]
    if len(hybrid_metrics) > 0:
        hybrid_x = np.arange(len(hybrid_metrics))
        ax.bar(hybrid_x, hybrid_metrics['avg_ml_confidence'], width, capsize=5)
        ax.set_ylabel('Average ML Confidence')
        ax.set_title('ML Model Contribution')
        ax.set_xticks(hybrid_x)
        ax.set_xticklabels(hybrid_metrics['scheduler'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'metrics_comparison.png'}")

def plot_learning_curves(results_df: pd.DataFrame, output_dir: Path):
    """Plot learning curves over review number."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    schedulers = results_df['scheduler'].unique()
    colors = sns.color_palette("husl", len(schedulers))
    
    # 1. Average interval over reviews
    ax = axes[0, 0]
    for i, sched in enumerate(schedulers):
        subset = results_df[results_df['scheduler'] == sched]
        grouped = subset.groupby('review_num')['interval'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=sched, 
               color=colors[i], linewidth=2, markersize=4)
    ax.set_xlabel('Review Number')
    ax.set_ylabel('Average Interval (days)')
    ax.set_title('Interval Growth Over Reviews')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average half-life over reviews
    ax = axes[0, 1]
    for i, sched in enumerate(schedulers):
        subset = results_df[results_df['scheduler'] == sched]
        grouped = subset.groupby('review_num')['halflife'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=sched,
               color=colors[i], linewidth=2, markersize=4)
    ax.set_xlabel('Review Number')
    ax.set_ylabel('Average Half-life (days)')
    ax.set_title('Memory Half-life Growth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Prediction accuracy over reviews
    ax = axes[1, 0]
    for i, sched in enumerate(schedulers):
        subset = results_df[results_df['scheduler'] == sched]
        subset['error'] = np.abs(subset['p_recall_pred'] - subset['p_recall_actual'])
        grouped = subset.groupby('review_num')['error'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=sched,
               color=colors[i], linewidth=2, markersize=4)
    ax.set_xlabel('Review Number')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Accuracy Over Reviews')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Average predicted recall over reviews
    ax = axes[1, 1]
    for i, sched in enumerate(schedulers):
        subset = results_df[results_df['scheduler'] == sched]
        grouped = subset.groupby('review_num')['p_recall_pred'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=sched,
               color=colors[i], linewidth=2, markersize=4)
    ax.set_xlabel('Review Number')
    ax.set_ylabel('Average Predicted Recall')
    ax.set_title('Predicted Recall Over Reviews')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'learning_curves.png'}")

def create_all_visualizations(results_path: str, metrics_path: str, output_dir: str = "./visualizations"):
    """Create all visualizations."""
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    results_df, metrics_df = load_comparison_data(results_path, metrics_path)
    
    print(f"\nLoaded {len(results_df)} results")
    print(f"Schedulers: {', '.join(results_df['scheduler'].unique())}")
    
    # Create visualizations
    print("\n[1/4] Creating prediction accuracy plots...")
    plot_prediction_accuracy(results_df, output_dir)
    
    print("\n[2/4] Creating ML contribution plots...")
    plot_ml_contribution(results_df, output_dir)
    
    print("\n[3/4] Creating metrics comparison...")
    plot_metrics_comparison(metrics_df, output_dir)
    
    print("\n[4/4] Creating learning curves...")
    plot_learning_curves(results_df, output_dir)
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS CREATED!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.glob("*.png"):
        print(f"  - {f.name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create research visualizations')
    parser.add_argument('results_csv', help='Comparison results CSV')
    parser.add_argument('metrics_csv', help='Metrics CSV')
    parser.add_argument('--output-dir', default='./visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    create_all_visualizations(args.results_csv, args.metrics_csv, args.output_dir)

