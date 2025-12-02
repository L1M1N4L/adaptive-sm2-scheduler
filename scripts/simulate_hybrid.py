"""
Simulate learning with hybrid scheduler using SSP-MMC-Plus framework
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schedulers.hybrid import HybridScheduler
from src.schedulers.base import RatingConverter

def simulate_hybrid_scheduler(
    data_path: str,
    hlr_weights_path: str = None,
    dhp_params_path: str = None,
    max_cards: int = 1000,
    output_path: str = "./simulation/hybrid_results.tsv"
):
    """
    Simulate learning with hybrid scheduler.
    
    Similar to simulator.py but uses our hybrid scheduler.
    """
    print("=" * 60)
    print("Hybrid Scheduler Simulation")
    print("=" * 60)
    
    # Initialize hybrid scheduler with trained models
    print("\nInitializing hybrid scheduler...")
    hybrid = HybridScheduler(
        use_hlr=True,
        use_dhp=True,
        use_rnn=False,
        hlr_weights_path=hlr_weights_path,
        dhp_params_path=dhp_params_path
    )
    
    print(f"HLR available: {hybrid.use_hlr}")
    print(f"DHP available: {hybrid.use_dhp}")
    
    # Load training data format
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, sep='\t')
    
    # Filter valid data
    df = df[df['halflife'] > 0]
    df = df[df['i'] > 0]
    
    if max_cards:
        # Sample cards
        unique_difficulties = df['d'].unique()
        sampled_df = pd.DataFrame()
        for d in unique_difficulties[:max_cards//10]:
            subset = df[df['d'] == d]
            if len(subset) > 0:
                sampled_df = pd.concat([sampled_df, subset.sample(min(100, len(subset)), random_state=2022)])
        df = sampled_df
    
    print(f"Processing {len(df)} samples...")
    
    # Simulation parameters
    learn_days = 365
    target_halflife = 180
    
    # Track metrics
    metrics = {
        'day': [],
        'total_reviews': [],
        'total_learned': [],
        'avg_halflife': [],
        'avg_p_recall': [],
        'meet_target': [],
        'ml_confidence_avg': []
    }
    
    # Process each sample as a separate learning item
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Parse history
        r_history = str(row['r_history']).split(',')
        t_history = str(row['t_history']).split(',')
        p_history = str(row['p_history']).split(',')
        
        user_id = "sim_user"
        item_id = f"item_{idx}"
        
        # Simulate reviews
        current_time = 0.0
        review_count = 0
        total_ml_confidence = 0.0
        
        for i, (r, t, p) in enumerate(zip(r_history, t_history, p_history)):
            if i == 0:
                continue  # Skip first (no history yet)
            
            # Convert recall to rating
            recall = int(r)
            # Use SM-2 quality scale: 4 for correct, 0 for wrong
            rating = 4 if recall == 1 else 0
            
            # Update time
            interval = int(float(t))
            current_time += interval
            
            # Schedule review
            try:
                decision = hybrid.schedule_review(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    timestamp=current_time
                )
                
                review_count += 1
                total_ml_confidence += decision.confidence
                
                # Get half-life
                halflife = hybrid.calculate_half_life(user_id, item_id)
                
                results.append({
                    'item_id': item_id,
                    'review_num': i,
                    'day': current_time,
                    'rating': rating,
                    'interval': decision.interval,
                    'halflife': halflife,
                    'p_recall': decision.p_recall,
                    'ml_confidence': decision.confidence,
                    'ease_factor': decision.ease_factor,
                    'repetitions': decision.repetitions
                })
                
            except Exception as e:
                print(f"Error processing item {item_id}, review {i}: {e}")
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate daily metrics
    for day in range(1, min(learn_days + 1, int(results_df['day'].max()) + 1)):
        day_data = results_df[results_df['day'] <= day]
        
        metrics['day'].append(day)
        metrics['total_reviews'].append(len(day_data))
        metrics['total_learned'].append(day_data['item_id'].nunique())
        metrics['avg_halflife'].append(day_data['halflife'].mean() if len(day_data) > 0 else 0)
        metrics['avg_p_recall'].append(day_data['p_recall'].mean() if len(day_data) > 0 else 0)
        metrics['meet_target'].append((day_data['halflife'] >= target_halflife).sum() if len(day_data) > 0 else 0)
        metrics['ml_confidence_avg'].append(day_data['ml_confidence'].mean() if len(day_data) > 0 else 0)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, sep='\t', index=False)
    
    metrics_path = output_path.parent / "hybrid_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep='\t', index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {output_path}")
    print(f"  - {metrics_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Simulation Summary")
    print("=" * 60)
    print(f"Total reviews: {len(results_df)}")
    print(f"Total items: {results_df['item_id'].nunique()}")
    print(f"Average ML confidence: {results_df['ml_confidence'].mean():.2%}")
    print(f"Reviews with ML contribution: {(results_df['ml_confidence'] > 0).sum()} ({(results_df['ml_confidence'] > 0).sum() / len(results_df) * 100:.1f}%)")
    print(f"Average half-life: {results_df['halflife'].mean():.2f} days")
    print(f"Items meeting target ({target_halflife}d): {(results_df['halflife'] >= target_halflife).sum()}")
    
    return results_df, metrics_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate learning with hybrid scheduler')
    parser.add_argument('data_tsv', help='Training data in SSP-MMC-Plus TSV format')
    parser.add_argument('--hlr-weights', help='Path to trained HLR weights')
    parser.add_argument('--dhp-params', help='Path to trained DHP parameters')
    parser.add_argument('--max-cards', type=int, default=1000, help='Maximum cards to simulate')
    parser.add_argument('--output', default='./simulation/hybrid_results.tsv', help='Output path')
    
    args = parser.parse_args()
    
    simulate_hybrid_scheduler(
        args.data_tsv,
        hlr_weights_path=args.hlr_weights,
        dhp_params_path=args.dhp_params,
        max_cards=args.max_cards,
        output_path=args.output
    )

