"""
Fair comparison of schedulers - ensures all use same data and methodology
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schedulers.hybrid import HybridScheduler
from src.schedulers.sm2 import SM2Scheduler

def evaluate_scheduler_fair(scheduler, data_df: pd.DataFrame, scheduler_name: str):
    """
    Evaluate scheduler on the SAME dataset for fair comparison.
    
    Args:
        scheduler: Scheduler instance
        data_df: DataFrame with evaluation data (same for all schedulers)
        scheduler_name: Name of scheduler
    """
    print(f"\nEvaluating {scheduler_name}...")
    
    results = []
    
    # Reset scheduler state for fair comparison
    if hasattr(scheduler, 'reset'):
        scheduler.reset()
    elif hasattr(scheduler, 'reset_state'):
        scheduler.reset_state()
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=scheduler_name):
        # Parse history
        r_history = str(row['r_history']).split(',')
        t_history = str(row['t_history']).split(',')
        p_history = str(row['p_history']).split(',')
        
        user_id = "eval_user"
        item_id = f"item_{idx}"
        
        # Simulate reviews - same sequence for all schedulers
        current_time = 0.0
        
        for i, (r, t, p) in enumerate(zip(r_history, t_history, p_history)):
            if i == 0:
                continue
            
            # Convert recall to rating (same conversion for all)
            recall = int(r)
            rating = 4 if recall == 1 else 0
            
            # Update time (same timeline for all)
            interval = int(float(t))
            current_time += interval
            
            try:
                decision = scheduler.schedule_review(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    timestamp=current_time
                )
                
                halflife = scheduler.calculate_half_life(user_id, item_id)
                p_recall_pred = scheduler.predict_recall(user_id, item_id, delta_t=interval)
                
                results.append({
                    'scheduler': scheduler_name,
                    'item_id': item_id,
                    'review_num': i,
                    'interval': decision.interval,
                    'halflife': halflife,
                    'p_recall_pred': p_recall_pred,
                    'p_recall_actual': float(p),
                    'ml_confidence': getattr(decision, 'confidence', 0.0),
                    'ease_factor': decision.ease_factor,
                    'repetitions': decision.repetitions,
                    'timestamp': current_time
                })
                
            except Exception as e:
                # Log error but continue
                continue
    
    return pd.DataFrame(results)

def compare_all_schedulers_fair(data_path: str, 
                                hlr_weights_path: str = None,
                                dhp_params_path: str = None,
                                max_samples: int = 1000,
                                output_path: str = "./comparison_results.csv",
                                random_seed: int = 2022):
    """
    Fair comparison ensuring all schedulers use identical data.
    """
    print("=" * 80)
    print("FAIR SCHEDULER COMPARISON")
    print("=" * 80)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load and prepare data ONCE
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, sep='\t')
    df = df[df['halflife'] > 0]
    df = df[df['i'] > 0]
    
    # Sample data ONCE - same for all schedulers
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_seed)
        df = df.sort_index()  # Ensure consistent ordering
    
    print(f"Using {len(df)} samples for evaluation")
    print(f"Total review events: {len(df)}")
    
    all_results = []
    
    # 1. SM-2 Baseline
    print("\n[1/3] SM-2 Baseline")
    sm2 = SM2Scheduler()
    sm2_results = evaluate_scheduler_fair(sm2, df, "SM-2")
    all_results.append(sm2_results)
    
    # 2. Hybrid (with trained models)
    print("\n[2/3] Hybrid (SM-2 + ML)")
    hybrid = HybridScheduler(
        use_hlr=True,
        use_dhp=True,
        use_rnn=False,
        hlr_weights_path=hlr_weights_path,
        dhp_params_path=dhp_params_path,
        pure_ml=False
    )
    hybrid_results = evaluate_scheduler_fair(hybrid, df, "Hybrid")
    all_results.append(hybrid_results)
    
    # 3. Pure ML (ML-only, no SM-2 blending)
    print("\n[3/3] Pure ML (ML-only)")
    pure_ml = HybridScheduler(
        use_hlr=True,
        use_dhp=True,
        use_rnn=False,
        hlr_weights_path=hlr_weights_path,
        dhp_params_path=dhp_params_path,
        pure_ml=True  # Use only ML predictions
    )
    pure_ml_results = evaluate_scheduler_fair(pure_ml, df, "Pure-ML")
    all_results.append(pure_ml_results)
    
    # Combine results
    comparison_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("CALCULATING METRICS")
    print("=" * 80)
    
    metrics = []
    for scheduler_name in comparison_df['scheduler'].unique():
        subset = comparison_df[comparison_df['scheduler'] == scheduler_name]
        
        # Prediction accuracy
        mae_p = np.mean(np.abs(subset['p_recall_pred'] - subset['p_recall_actual']))
        mse_p = np.mean((subset['p_recall_pred'] - subset['p_recall_actual']) ** 2)
        rmse_p = np.sqrt(mse_p)
        
        # Average metrics
        avg_interval = subset['interval'].mean()
        avg_halflife = subset['halflife'].mean()
        avg_p_recall = subset['p_recall_pred'].mean()
        avg_ml_confidence = subset['ml_confidence'].mean() if 'ml_confidence' in subset.columns else 0.0
        
        # Additional metrics
        median_interval = subset['interval'].median()
        std_interval = subset['interval'].std()
        
        metrics.append({
            'scheduler': scheduler_name,
            'mae_p': mae_p,
            'mse_p': mse_p,
            'rmse_p': rmse_p,
            'avg_interval': avg_interval,
            'median_interval': median_interval,
            'std_interval': std_interval,
            'avg_halflife': avg_halflife,
            'avg_p_recall': avg_p_recall,
            'avg_ml_confidence': avg_ml_confidence,
            'n_samples': len(subset),
            'n_items': subset['item_id'].nunique()
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    
    metrics_path = output_path.parent / "comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    
    print(f"\nDetailed results saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")
    
    return comparison_df, metrics_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fair comparison of schedulers')
    parser.add_argument('data_tsv', help='Evaluation data TSV file (SSP-MMC-Plus format)')
    parser.add_argument('--hlr-weights', help='Path to HLR weights')
    parser.add_argument('--dhp-params', help='Path to DHP parameters')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples to evaluate')
    parser.add_argument('--output', default='./research_output/comparison_results.csv', help='Output path')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    compare_all_schedulers_fair(
        args.data_tsv,
        hlr_weights_path=args.hlr_weights,
        dhp_params_path=args.dhp_params,
        max_samples=args.max_samples,
        output_path=args.output,
        random_seed=args.seed
    )

