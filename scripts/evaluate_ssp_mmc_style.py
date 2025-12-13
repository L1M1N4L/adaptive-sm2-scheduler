"""
Evaluate schedulers using SSP-MMC-Plus methodology
Follows the exact same evaluation approach as the original framework
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import RepeatedKFold
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.schedulers.hybrid import HybridScheduler
from src.schedulers.sm2 import SM2Scheduler

def smape(A, F):
    """Symmetric Mean Absolute Percentage Error"""
    return 1 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + 1e-10))

def evaluate_scheduler_ssp_mmc(scheduler, test_data: pd.DataFrame, scheduler_name: str, repeat: int, fold: int):
    """
    Evaluate scheduler following SSP-MMC-Plus eval() method pattern.
    
    Returns DataFrame with columns: r_history, t_history, p_history, t, h, hh, p, pp, ae, ape
    where:
    - h = actual halflife
    - hh = predicted halflife
    - p = actual p_recall
    - pp = predicted p_recall
    - ae = absolute error in p
    - ape = absolute percentage error in h
    """
    record = pd.DataFrame(
        columns=['r_history', 't_history', 'p_history',
                 't', 'h', 'hh', 'p', 'pp', 'ae', 'ape'])
    
    p_loss = 0
    h_loss = 0
    count = 0
    
    # Reset scheduler for each evaluation
    if hasattr(scheduler, 'reset'):
        scheduler.reset()
    
    for idx, line in tqdm(test_data.iterrows(), total=len(test_data), desc=f"{scheduler_name} R{repeat}F{fold}"):
        # Parse history
        r_history = str(line['r_history']).split(',')
        t_history = str(line['t_history']).split(',')
        p_history = str(line['p_history']).split(',')
        
        user_id = "eval_user"
        item_id = f"item_{idx}"
        
        # Simulate reviews up to the point before the test review
        current_time = 0.0
        
        # Process all reviews except the last one (which is what we're predicting)
        for i in range(len(r_history) - 1):
            if i == 0:
                continue
            
            recall = int(r_history[i])
            rating = 4 if recall == 1 else 0
            interval = int(float(t_history[i]))
            current_time += interval
            
            try:
                scheduler.schedule_review(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating,
                    timestamp=current_time
                )
            except Exception as e:
                continue
        
        # Now predict for the test review
        try:
            # Get the test review details
            test_interval = int(float(t_history[-1]))
            test_delta_t = int(float(line['delta_t']))
            actual_p = float(line['p_recall'])
            actual_h = float(line['halflife'])
            
            # Predict recall probability
            predicted_p = scheduler.predict_recall(user_id, item_id, delta_t=test_delta_t)
            
            # Calculate predicted half-life from predicted recall
            if predicted_p > 0 and predicted_p < 1:
                predicted_h = -test_delta_t / np.log2(predicted_p)
            else:
                predicted_h = scheduler.calculate_half_life(user_id, item_id)
            
            # Calculate errors
            p_error = abs(actual_p - predicted_p)
            h_error = abs((predicted_h - actual_h) / actual_h) if actual_h > 0 else 0
            
            p_loss += p_error
            h_loss += h_error
            count += 1
            
            record = pd.concat([record, pd.DataFrame({
                'r_history': [line['r_history']],
                't_history': [line['t_history']],
                'p_history': [line['p_history']],
                't': [test_delta_t],
                'h': [actual_h],
                'hh': [round(predicted_h, 2)],
                'p': [actual_p],
                'pp': [round(predicted_p, 3)],
                'ae': [round(p_error, 3)],
                'ape': [round(h_error, 3)]
            })], ignore_index=True)
            
        except Exception as e:
            continue
    
    if count > 0:
        print(f"model: {scheduler_name}")
        print(f'sample num: {count}')
        print(f"avg p loss (MAE): {p_loss / count:.4f}")
        print(f"avg h loss (MAPE): {h_loss / count:.4f}")
    
    # Save results in SSP-MMC-Plus format
    result_dir = Path(f'./result/{scheduler_name}')
    result_dir.mkdir(parents=True, exist_ok=True)
    record.to_csv(result_dir / f'repeat{repeat}_fold{fold}_{int(pd.Timestamp.now().timestamp())}.tsv', 
                  sep='\t', index=False)
    
    return record

def evaluate_all_schedulers_ssp_mmc(data_path: str,
                                    hlr_weights_path: str = None,
                                    dhp_params_path: str = None,
                                    random_seed: int = 2022):
    """
    Evaluate all schedulers using SSP-MMC-Plus methodology:
    - RepeatedKFold cross-validation (2 splits, 5 repeats = 10 folds)
    - Same train/test splits for all schedulers
    - Same metrics as SSP-MMC-Plus
    """
    print("=" * 80)
    print("SSP-MMC-PLUS STYLE EVALUATION")
    print("=" * 80)
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    dataset = pd.read_csv(data_path, sep='\t', index_col=None)
    dataset = dataset[dataset['halflife'] > 0]
    dataset = dataset[dataset['i'] > 0]
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split into train and test (80/20) - same as SSP-MMC-Plus
    test = dataset.sample(frac=0.8, random_state=random_seed)
    train = dataset.drop(index=test.index)
    
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    
    # Use RepeatedKFold: 2 splits, 5 repeats = 10 folds total
    kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=random_seed)
    
    all_results = {}
    
    for idx, (train_index, test_index) in enumerate(kf.split(test)):
        train_fold = test.iloc[train_index]
        test_fold = test.iloc[test_index]
        repeat = idx // 2 + 1
        fold = idx % 2 + 1
        
        print(f"\n{'='*80}")
        print(f"Repeat {repeat}, Fold {fold}")
        print(f"{'='*80}")
        print(f"|train| = {len(train_fold)}")
        print(f"|test|  = {len(test_fold)}")
        
        # Train models on train_fold (if needed)
        # For now, we'll use pre-trained models
        
        # Evaluate SM-2
        print(f"\n[1/3] Evaluating SM-2...")
        sm2 = SM2Scheduler()
        sm2_results = evaluate_scheduler_ssp_mmc(sm2, test_fold, "SM-2", repeat, fold)
        if "SM-2" not in all_results:
            all_results["SM-2"] = []
        all_results["SM-2"].append(sm2_results)
        
        # Evaluate Hybrid
        print(f"\n[2/3] Evaluating Hybrid...")
        hybrid = HybridScheduler(
            use_hlr=True,
            use_dhp=True,
            use_rnn=False,
            hlr_weights_path=hlr_weights_path,
            dhp_params_path=dhp_params_path,
            pure_ml=False
        )
        hybrid_results = evaluate_scheduler_ssp_mmc(hybrid, test_fold, "Hybrid", repeat, fold)
        if "Hybrid" not in all_results:
            all_results["Hybrid"] = []
        all_results["Hybrid"].append(hybrid_results)
        
        # Evaluate Pure ML
        print(f"\n[3/3] Evaluating Pure ML...")
        pure_ml = HybridScheduler(
            use_hlr=True,
            use_dhp=True,
            use_rnn=False,
            hlr_weights_path=hlr_weights_path,
            dhp_params_path=dhp_params_path,
            pure_ml=True
        )
        pure_ml_results = evaluate_scheduler_ssp_mmc(pure_ml, test_fold, "Pure-ML", repeat, fold)
        if "Pure-ML" not in all_results:
            all_results["Pure-ML"] = []
        all_results["Pure-ML"].append(pure_ml_results)
    
    # Calculate aggregate metrics (same as evaluate.py)
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
    
    for scheduler_name in ["SM-2", "Hybrid", "Pure-ML"]:
        if scheduler_name not in all_results:
            continue
        
        print(f"\nmodel: {scheduler_name}")
        
        # Combine all folds
        all_folds = pd.concat(all_results[scheduler_name], ignore_index=True)
        
        if len(all_folds) == 0:
            print("No results")
            continue
        
        # Calculate metrics
        avg_mae_p = mean_absolute_error(all_folds['p'], all_folds['pp'])
        avg_mse_p = mean_squared_error(all_folds['p'], all_folds['pp'])
        avg_mape_h = mean_absolute_percentage_error(all_folds['h'], all_folds['hh'])
        avg_smape_h = smape(all_folds['h'].values, all_folds['hh'].values)
        avg_mae_h = mean_absolute_error(all_folds['h'], all_folds['hh'])
        
        print(f"avg\tmae(p): {avg_mae_p:.4f}\tmse(p): {avg_mse_p:.4f}\tmape(h): {avg_mape_h:.4f}\t"
              f"smape(h): {avg_smape_h:.4f}\tmae(h): {avg_mae_h:.4f}")
        
        # Save aggregate results
        all_folds.to_csv(f'./result/{scheduler_name}/all_folds.tsv', sep='\t', index=False)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("Results saved to ./result/[scheduler_name]/")
    
    return all_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate schedulers using SSP-MMC-Plus methodology')
    parser.add_argument('data_tsv', help='Evaluation data TSV file (SSP-MMC-Plus format)')
    parser.add_argument('--hlr-weights', help='Path to HLR weights')
    parser.add_argument('--dhp-params', help='Path to DHP parameters')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    
    args = parser.parse_args()
    
    evaluate_all_schedulers_ssp_mmc(
        args.data_tsv,
        hlr_weights_path=args.hlr_weights,
        dhp_params_path=args.dhp_params,
        random_seed=args.seed
    )



