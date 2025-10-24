"""
SSP-MMC-Plus Integration Module

This module integrates our professional SM-2 implementation with the SSP-MMC-Plus
evaluation framework for comprehensive spaced repetition algorithm comparison.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add our src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.schedulers import SM2Scheduler, RatingConverter


class SM2SSPMMCAdapter:
    """
    Adapter class to integrate our SM-2 implementation with SSP-MMC-Plus evaluation.
    
    This class converts between our SM-2 interface and the SSP-MMC-Plus format
    for comprehensive evaluation using the established metrics.
    """
    
    def __init__(self):
        """Initialize the adapter with our SM-2 scheduler."""
        self.scheduler = SM2Scheduler()
        self.results = []
    
    def convert_ssp_mmc_to_sm2(self, ssp_data):
        """
        Convert SSP-MMC-Plus data format to our SM-2 format.
        
        Args:
            ssp_data: DataFrame with SSP-MMC-Plus columns
            
        Returns:
            DataFrame with converted data
        """
        converted_data = ssp_data.copy()
        
        # Convert recall results to SM-2 quality scale
        # SSP-MMC-Plus uses: 1=correct, 2=incorrect, 3=incorrect
        # Our SM-2 uses: 0-5 scale
        def convert_recall_to_quality(recall):
            if recall == 1:  # Correct
                return 4  # Good
            else:  # Incorrect (2 or 3)
                return 2  # Incorrect but remembered
        
        converted_data['sm2_quality'] = converted_data['r'].apply(convert_recall_to_quality)
        
        return converted_data
    
    def evaluate_sm2_on_ssp_mmc(self, testset, repeat, fold):
        """
        Evaluate our SM-2 implementation using SSP-MMC-Plus evaluation framework.
        
        Args:
            testset: SSP-MMC-Plus test dataset
            repeat: Repeat number for evaluation
            fold: Fold number for evaluation
            
        Returns:
            DataFrame with evaluation results
        """
        print(f"Evaluating our SM-2 implementation on SSP-MMC-Plus dataset...")
        print(f"Repeat: {repeat}, Fold: {fold}")
        
        # Convert data format
        converted_data = self.convert_ssp_mmc_to_sm2(testset)
        
        # Initialize results DataFrame
        record = pd.DataFrame(columns=[
            'r_history', 't_history', 'p_history',
            't', 'h', 'hh', 'p', 'pp', 'loss'
        ])
        
        total_loss = 0
        count = 0
        
        # Process each sample
        for idx, line in converted_data.iterrows():
            try:
                # Extract user and item identifiers
                user_id = f"user_{line['u']}"
                item_id = f"item_{line['w']}"
                
                # Don't reset scheduler - we want to maintain state across items
                
                # Process historical reviews if available
                if pd.notna(line['r_history']) and pd.notna(line['t_history']):
                    r_history = eval(line['r_history']) if isinstance(line['r_history'], str) else line['r_history']
                    t_history = eval(line['t_history']) if isinstance(line['t_history'], str) else line['t_history']
                    
                    # Process historical reviews
                    for r, t in zip(r_history, t_history):
                        sm2_quality = 4 if r == 1 else 2  # Convert to SM-2 scale
                        self.scheduler.schedule_review(
                            user_id=user_id,
                            item_id=item_id,
                            rating=sm2_quality,
                            timestamp=t
                        )
                
                # Process current review
                current_quality = line['sm2_quality']
                current_timestamp = line['delta_t']
                
                decision = self.scheduler.schedule_review(
                    user_id=user_id,
                    item_id=item_id,
                    rating=current_quality,
                    timestamp=current_timestamp
                )
                
                # Calculate predicted recall probability
                pp = decision.p_recall
                p = line['p']  # Actual recall probability from dataset
                
                # Calculate half-life
                hh = self.scheduler.calculate_half_life(user_id, item_id)
                h = line['h']  # Actual half-life from dataset
                
                # Calculate loss
                loss = abs(p - pp)
                total_loss += loss
                count += 1
                
                # Store results
                record = pd.concat([record, pd.DataFrame({
                    'r_history': [line['r_history']],
                    't_history': [line['t_history']],
                    'p_history': [line.get('p_history', '[]')],
                    't': [line['delta_t']],  # Use delta_t as t
                    'h': [h],
                    'hh': [round(hh, 2)],
                    'p': [p],
                    'pp': [round(pp, 3)],
                    'loss': [round(loss, 3)]
                })], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        # Print evaluation results
        print(f"Model: Our SM-2 Implementation")
        print(f"Sample num: {count}")
        print(f"Avg loss: {total_loss / count if count > 0 else 0:.4f}")
        
        # Save results
        Path('./evaluation/results/our_sm2').mkdir(parents=True, exist_ok=True)
        result_file = f'./evaluation/results/our_sm2/repeat{repeat}_fold{fold}_{int(time.time())}.tsv'
        record.to_csv(result_file, sep='\t', index=False)
        print(f"Results saved to: {result_file}")
        
        return record
    
    def compare_with_baseline_sm2(self, testset, repeat, fold):
        """
        Compare our SM-2 implementation with the baseline SM-2 in SSP-MMC-Plus.
        
        Args:
            testset: SSP-MMC-Plus test dataset
            repeat: Repeat number for evaluation
            fold: Fold number for evaluation
            
        Returns:
            Comparison results
        """
        print("Comparing our SM-2 with baseline SM-2...")
        
        # Evaluate our implementation
        our_results = self.evaluate_sm2_on_ssp_mmc(testset, repeat, fold)
        
        # Calculate metrics
        our_mae = np.mean(np.abs(our_results['p'] - our_results['pp']))
        our_mae_h = np.mean(np.abs(our_results['h'] - our_results['hh']))
        
        print(f"Our SM-2 - MAE(p): {our_mae:.4f}, MAE(h): {our_mae_h:.4f}")
        
        return {
            'our_sm2': {
                'mae_p': our_mae,
                'mae_h': our_mae_h,
                'samples': len(our_results)
            }
        }


def run_ssp_mmc_evaluation(data_path, output_dir="./evaluation/results"):
    """
    Run complete SSP-MMC-Plus evaluation with our SM-2 implementation.
    
    Args:
        data_path: Path to SSP-MMC-Plus dataset
        output_dir: Directory to save results
    """
    print("Starting SSP-MMC-Plus evaluation with our SM-2 implementation...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize adapter
    adapter = SM2SSPMMCAdapter()
    
    # Load dataset (assuming it's already preprocessed)
    # This would need to be adapted based on the actual data format
    print("Note: This is a template for integration.")
    print("The actual dataset needs to be loaded and the data format adapted.")
    
    return adapter


if __name__ == "__main__":
    print("SSP-MMC-Plus Integration Module")
    print("=" * 40)
    
    # Test the adapter
    adapter = SM2SSPMMCAdapter()
    print("Adapter initialized successfully!")
    print("Ready for SSP-MMC-Plus evaluation integration.")
    
    # Example usage (when actual data is available)
    print("\nTo use with actual data:")
    print("1. Load the SSP-MMC-Plus dataset")
    print("2. Call adapter.evaluate_sm2_on_ssp_mmc(testset, repeat, fold)")
    print("3. Compare results with baseline algorithms")
