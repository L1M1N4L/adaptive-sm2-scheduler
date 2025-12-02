"""
Visualize hybrid scheduler metrics following SSP-MMC-Plus visualization style
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path

# Set style similar to visualization.py
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')
plt.rc('text', usetex=False)  # Set to False if LaTeX not available
plt.rc('font', family='serif')

# Camera settings for 3D plots
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=1.5, z=1.25)
)


def calculate_cumulative_metrics(df):
    """
    Calculate cumulative metrics over time for each scheduler.
    
    Returns:
        metrics_df: DataFrame with daily metrics
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm not available
        def tqdm(iterable, desc="", leave=False):
            return iterable
    
    metrics_list = []
    
    # Key days to sample
    key_days = [1, 7, 14, 30, 60, 90, 120, 180, 240, 300, 365]
    
    for scheduler in df['scheduler'].unique():
        print(f"  Processing {scheduler}...")
        subset = df[df['scheduler'] == scheduler].copy()
        
        # Calculate cumulative day for each review
        subset = subset.sort_values(['item_id', 'review_num'])
        subset['cumulative_day'] = subset.groupby('item_id')['interval'].cumsum()
        
        # Get max day to iterate through
        max_day = int(subset['cumulative_day'].max()) if len(subset) > 0 else 365
        max_day = min(max_day, 365)  # Cap at 365 days
        
        # Sample days (every 10 days for efficiency, plus key days)
        sample_days = sorted(set(list(range(1, max_day + 1, 10)) + [d for d in key_days if d <= max_day]))
        
        # Pre-compute item states for efficiency
        item_states = {}
        for item_id in subset['item_id'].unique():
            item_data = subset[subset['item_id'] == item_id].sort_values('review_num')
            item_states[item_id] = {
                'days': item_data['cumulative_day'].values,
                'recalls': item_data['p_recall_pred'].values,
                'halflives': item_data['halflife'].values,
                'review_nums': item_data['review_num'].values
            }
        
        for day in tqdm(sample_days, desc=f"    {scheduler} days", leave=False):
            # Collect all reviews up to this day
            day_recalls = []
            day_items_state = {}  # item_id -> latest state
            
            for item_id, states in item_states.items():
                # Find reviews up to this day
                mask = states['days'] <= day
                if not mask.any():
                    continue
                
                # Get all reviews for this item up to this day
                item_recalls = states['recalls'][mask]
                day_recalls.extend(item_recalls)
                
                # Get latest state
                latest_idx = np.where(mask)[0][-1]
                day_items_state[item_id] = {
                    'halflife': states['halflives'][latest_idx],
                    'recalls': item_recalls,
                    'review_nums': states['review_nums'][mask]
                }
            
            if len(day_recalls) == 0:
                continue
            
            # Calculate metrics
            # 1. Average recall probability
            avg_recall = np.mean(day_recalls)
            
            # 2. THR: % of items with half-life >= 180 or 365
            halflives = [state['halflife'] for state in day_items_state.values()]
            total_items = len(halflives)
            if total_items > 0:
                thr_180 = sum(1 for h in halflives if h >= 180) / total_items * 100
                thr_365 = sum(1 for h in halflives if h >= 365) / total_items * 100
            else:
                thr_180 = 0
                thr_365 = 0
            
            # 3. SRP: Summation of Recall Probability
            srp = sum(day_recalls)
            
            # 4. WTL: items with recall >= 90% for last three reviews
            wtl_items = 0
            for item_id, state in day_items_state.items():
                if len(state['recalls']) >= 3:
                    last_three = state['recalls'][-3:]
                    if all(r >= 0.90 for r in last_three):
                        wtl_items += 1
            wtl = wtl_items
            
            # 5. Daily Cost: average reviews/day (total reviews / days)
            daily_cost = len(day_recalls) / day if day > 0 else 0
            
            # 6. Efficiency = SRP / Cost (average recall per review)
            efficiency = avg_recall  # This is already the average
            
            metrics_list.append({
                'scheduler': scheduler,
                'day': day,
                'avg_recall': avg_recall,
                'thr_180': thr_180,
                'thr_365': thr_365,
                'srp': srp,
                'wtl': wtl,
                'daily_cost': daily_cost,
                'efficiency': efficiency,
                'total_reviews': len(day_recalls),
                'total_items': total_items
            })
    
    return pd.DataFrame(metrics_list)


def plot_recall_curves(metrics_df, output_dir):
    """
    Plot recall curves: average recall probability at 30, 60, 180, 365 days.
    """
    fig = go.Figure()
    
    target_days = [30, 60, 180, 365]
    schedulers = metrics_df['scheduler'].unique()
    colors = px.colors.qualitative.Set2[:len(schedulers)]
    
    # Plot curves for each scheduler
    for i, scheduler in enumerate(schedulers):
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        fig.add_trace(go.Scatter(
            x=scheduler_data['day'],
            y=scheduler_data['avg_recall'],
            mode='lines',
            name=scheduler,
            line=dict(color=colors[i], width=2)
        ))
    
    # Mark target days
    for day in target_days:
        fig.add_vline(
            x=day,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{day}d",
            annotation_position="top"
        )
    
    # Extract values at target days
    target_values = []
    for scheduler in schedulers:
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        values = []
        for day in target_days:
            day_data = scheduler_data[scheduler_data['day'] <= day]
            if len(day_data) > 0:
                values.append(day_data.iloc[-1]['avg_recall'])
            else:
                values.append(0)
        target_values.append(values)
    
    # Add table with target day values
    fig.add_trace(go.Table(
        header=dict(
            values=['Scheduler'] + [f'{d}d' for d in target_days],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=14)
        ),
        cells=dict(
            values=[[s] + [f'{v:.3f}' for v in vals] for s, vals in zip(schedulers, target_values)],
            fill_color='lavender',
            align='left',
            font=dict(size=12)
        ),
        domain=dict(x=[0.6, 1.0], y=[0.0, 0.3])
    ))
    
    fig.update_xaxes(
        title_text='Days',
        title_font=dict(size=24),
        tickfont=dict(size=18),
        range=[0, 365]
    )
    fig.update_yaxes(
        title_text='Average Recall Probability',
        title_font=dict(size=24),
        tickfont=dict(size=18),
        range=[0, 1]
    )
    fig.update_layout(
        title=dict(
            text='Recall Curves: Average Recall Probability Over Time',
            font=dict(size=26),
            x=0.5
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=80, b=50, l=80, r=200),
        width=1200,
        height=700
    )
    
    output_path = Path(output_dir) / 'recall_curves.png'
    try:
        fig.write_image(str(output_path), width=1200, height=700, engine='kaleido')
        print(f"Saved recall curves to {output_path}")
    except Exception as e:
        print(f"Error saving recall curves: {e}")
        fig.write_html(str(output_path).replace('.png', '.html'))
        print(f"Saved as HTML instead: {output_path}")


def plot_thr_metrics(metrics_df, output_dir):
    """
    Plot THR (Target Half-life Reached): % of items with half-life >= 180 or 365 days.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('THR at 180 days', 'THR at 365 days'),
        horizontal_spacing=0.15
    )
    
    schedulers = metrics_df['scheduler'].unique()
    colors = px.colors.qualitative.Set2[:len(schedulers)]
    
    for i, scheduler in enumerate(schedulers):
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        
        # THR 180
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['thr_180'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=True
            ),
            row=1, col=1
        )
        
        # THR 365
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['thr_365'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Update axes
    fig.update_xaxes(title_text='Days', title_font=dict(size=20), tickfont=dict(size=16), row=1, col=1)
    fig.update_xaxes(title_text='Days', title_font=dict(size=20), tickfont=dict(size=16), row=1, col=2)
    fig.update_yaxes(title_text='% Items', title_font=dict(size=20), tickfont=dict(size=16), row=1, col=1)
    fig.update_yaxes(title_text='% Items', title_font=dict(size=20), tickfont=dict(size=16), row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text='Target Half-life Reached (THR)',
            font=dict(size=26),
            x=0.5
        ),
        height=500,
        width=1400,
        margin=dict(t=80, b=50, l=80, r=50)
    )
    
    output_path = Path(output_dir) / 'thr_metrics.png'
    try:
        fig.write_image(str(output_path), width=1400, height=500, engine='kaleido')
        print(f"Saved THR metrics to {output_path}")
    except Exception as e:
        print(f"Error saving THR metrics: {e}")
        fig.write_html(str(output_path).replace('.png', '.html'))


def plot_srp_metrics(metrics_df, output_dir):
    """
    Plot SRP (Summation of Recall Probability) over time.
    """
    fig = go.Figure()
    
    schedulers = metrics_df['scheduler'].unique()
    colors = px.colors.qualitative.Set2[:len(schedulers)]
    
    for i, scheduler in enumerate(schedulers):
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        fig.add_trace(go.Scatter(
            x=scheduler_data['day'],
            y=scheduler_data['srp'],
            mode='lines',
            name=scheduler,
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_xaxes(
        title_text='Days',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_text='SRP (Summation of Recall Probability)',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_layout(
        title=dict(
            text='Summation of Recall Probability (SRP) Over Time',
            font=dict(size=26),
            x=0.5
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=80, b=50, l=100, r=50),
        width=1200,
        height=700
    )
    
    output_path = Path(output_dir) / 'srp_metrics.png'
    try:
        fig.write_image(str(output_path), width=1200, height=700, engine='kaleido')
        print(f"Saved SRP metrics to {output_path}")
    except Exception as e:
        print(f"Error saving SRP metrics: {e}")
        fig.write_html(str(output_path).replace('.png', '.html'))


def plot_wtl_metrics(metrics_df, output_dir):
    """
    Plot WTL (Words Total Learned): items with recall >= 90% for last three reviews.
    """
    fig = go.Figure()
    
    schedulers = metrics_df['scheduler'].unique()
    colors = px.colors.qualitative.Set2[:len(schedulers)]
    
    for i, scheduler in enumerate(schedulers):
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        fig.add_trace(go.Scatter(
            x=scheduler_data['day'],
            y=scheduler_data['wtl'],
            mode='lines',
            name=scheduler,
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_xaxes(
        title_text='Days',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_text='Words Total Learned (WTL)',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_layout(
        title=dict(
            text='Words Total Learned (WTL): Items with Recall ≥90% for Last 3 Reviews',
            font=dict(size=26),
            x=0.5
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=80, b=50, l=100, r=50),
        width=1200,
        height=700
    )
    
    output_path = Path(output_dir) / 'wtl_metrics.png'
    try:
        fig.write_image(str(output_path), width=1200, height=700, engine='kaleido')
        print(f"Saved WTL metrics to {output_path}")
    except Exception as e:
        print(f"Error saving WTL metrics: {e}")
        fig.write_html(str(output_path).replace('.png', '.html'))


def plot_daily_cost(metrics_df, output_dir):
    """
    Plot Daily Cost: average reviews/day.
    """
    fig = go.Figure()
    
    schedulers = metrics_df['scheduler'].unique()
    colors = px.colors.qualitative.Set2[:len(schedulers)]
    
    for i, scheduler in enumerate(schedulers):
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        fig.add_trace(go.Scatter(
            x=scheduler_data['day'],
            y=scheduler_data['daily_cost'],
            mode='lines',
            name=scheduler,
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_xaxes(
        title_text='Days',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_text='Daily Cost (Reviews/Day)',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_layout(
        title=dict(
            text='Daily Cost: Average Reviews per Day',
            font=dict(size=26),
            x=0.5
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=80, b=50, l=120, r=50),
        width=1200,
        height=700
    )
    
    output_path = Path(output_dir) / 'daily_cost.png'
    try:
        fig.write_image(str(output_path), width=1200, height=700, engine='kaleido')
        print(f"Saved daily cost to {output_path}")
    except Exception as e:
        print(f"Error saving daily cost: {e}")
        fig.write_html(str(output_path).replace('.png', '.html'))


def plot_efficiency(metrics_df, output_dir):
    """
    Plot Efficiency = SRP / Cost.
    """
    fig = go.Figure()
    
    schedulers = metrics_df['scheduler'].unique()
    colors = px.colors.qualitative.Set2[:len(schedulers)]
    
    for i, scheduler in enumerate(schedulers):
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        fig.add_trace(go.Scatter(
            x=scheduler_data['day'],
            y=scheduler_data['efficiency'],
            mode='lines',
            name=scheduler,
            line=dict(color=colors[i], width=2)
        ))
    
    fig.update_xaxes(
        title_text='Days',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_yaxes(
        title_text='Efficiency (SRP / Cost)',
        title_font=dict(size=24),
        tickfont=dict(size=18)
    )
    fig.update_layout(
        title=dict(
            text='Efficiency: SRP ÷ Cost',
            font=dict(size=26),
            x=0.5
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=80, b=50, l=120, r=50),
        width=1200,
        height=700
    )
    
    output_path = Path(output_dir) / 'efficiency.png'
    try:
        fig.write_image(str(output_path), width=1200, height=700, engine='kaleido')
        print(f"Saved efficiency to {output_path}")
    except Exception as e:
        print(f"Error saving efficiency: {e}")
        fig.write_html(str(output_path).replace('.png', '.html'))


def plot_comprehensive_comparison(metrics_df, output_dir):
    """
    Create a comprehensive comparison plot with all metrics.
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Recall Curves', 'THR (180 days)',
            'SRP', 'WTL',
            'Daily Cost', 'Efficiency'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    schedulers = metrics_df['scheduler'].unique()
    colors = px.colors.qualitative.Set2[:len(schedulers)]
    
    for i, scheduler in enumerate(schedulers):
        scheduler_data = metrics_df[metrics_df['scheduler'] == scheduler]
        
        # Recall curves
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['avg_recall'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=(i == 0)
            ),
            row=1, col=1
        )
        
        # THR 180
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['thr_180'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # SRP
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['srp'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # WTL
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['wtl'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Daily Cost
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['daily_cost'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Efficiency
        fig.add_trace(
            go.Scatter(
                x=scheduler_data['day'],
                y=scheduler_data['efficiency'],
                mode='lines',
                name=scheduler,
                line=dict(color=colors[i], width=2),
                legendgroup=scheduler,
                showlegend=False
            ),
            row=3, col=2
        )
    
    # Update axes labels
    fig.update_xaxes(title_text='Days', title_font=dict(size=16), tickfont=dict(size=12), row=1, col=1)
    fig.update_xaxes(title_text='Days', title_font=dict(size=16), tickfont=dict(size=12), row=1, col=2)
    fig.update_xaxes(title_text='Days', title_font=dict(size=16), tickfont=dict(size=12), row=2, col=1)
    fig.update_xaxes(title_text='Days', title_font=dict(size=16), tickfont=dict(size=12), row=2, col=2)
    fig.update_xaxes(title_text='Days', title_font=dict(size=16), tickfont=dict(size=12), row=3, col=1)
    fig.update_xaxes(title_text='Days', title_font=dict(size=16), tickfont=dict(size=12), row=3, col=2)
    
    fig.update_yaxes(title_text='Recall', title_font=dict(size=16), tickfont=dict(size=12), row=1, col=1)
    fig.update_yaxes(title_text='% Items', title_font=dict(size=16), tickfont=dict(size=12), row=1, col=2)
    fig.update_yaxes(title_text='SRP', title_font=dict(size=16), tickfont=dict(size=12), row=2, col=1)
    fig.update_yaxes(title_text='WTL', title_font=dict(size=16), tickfont=dict(size=12), row=2, col=2)
    fig.update_yaxes(title_text='Reviews/Day', title_font=dict(size=16), tickfont=dict(size=12), row=3, col=1)
    fig.update_yaxes(title_text='Efficiency', title_font=dict(size=16), tickfont=dict(size=12), row=3, col=2)
    
    fig.update_layout(
        title=dict(
            text='Comprehensive Hybrid Scheduler Metrics Comparison',
            font=dict(size=28),
            x=0.5
        ),
        height=1400,
        width=1600,
        margin=dict(t=100, b=50, l=80, r=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    output_path = Path(output_dir) / 'comprehensive_metrics.png'
    try:
        fig.write_image(str(output_path), width=1600, height=1400, engine='kaleido')
        print(f"Saved comprehensive comparison to {output_path}")
    except Exception as e:
        print(f"Error saving comprehensive metrics: {e}")
        fig.write_html(str(output_path).replace('.png', '.html'))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize hybrid scheduler metrics')
    parser.add_argument(
        'results_csv',
        help='Path to comparison_results.csv',
        default='research_output/comparison_results.csv'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for visualizations',
        default='research_output/visualizations'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading results from {args.results_csv}...")
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} rows with {df['scheduler'].nunique()} schedulers")
    
    # Calculate metrics
    print("Calculating cumulative metrics...")
    try:
        metrics_df = calculate_cumulative_metrics(df)
        print(f"Calculated metrics for {len(metrics_df)} time points")
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'hybrid_metrics_detailed.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved detailed metrics to {metrics_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_recall_curves(metrics_df, output_dir)
    plot_thr_metrics(metrics_df, output_dir)
    plot_srp_metrics(metrics_df, output_dir)
    plot_wtl_metrics(metrics_df, output_dir)
    plot_daily_cost(metrics_df, output_dir)
    plot_efficiency(metrics_df, output_dir)
    plot_comprehensive_comparison(metrics_df, output_dir)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()