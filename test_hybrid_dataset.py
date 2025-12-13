"""
Test Hybrid Scheduler with FSRS-Anki-20k dataset
"""

import pandas as pd
import numpy as np
from src.schedulers.hybrid import HybridScheduler
from src.schedulers.sm2 import SM2Scheduler
from src.schedulers.base import RatingConverter

def test_hybrid_with_dataset(csv_path: str, max_cards: int = 100):
    """Test hybrid scheduler with real dataset."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    # Initialize schedulers
    hybrid = HybridScheduler(use_hlr=True, use_dhp=True, use_rnn=False)
    sm2 = SM2Scheduler()
    
    # Group by card_id to process each card's review history
    card_ids = df['card_id'].unique()[:max_cards]
    print(f"\nProcessing {len(card_ids)} cards...")
    
    hybrid_results = []
    sm2_results = []
    
    for card_id in card_ids:
        card_data = df[df['card_id'] == card_id].sort_values('review_th')
        
        user_id = "test_user"
        item_id = f"card_{card_id}"
        
        # Track timestamps
        current_time = 0.0
        
        for idx, row in card_data.iterrows():
            # Convert Anki rating (1-4) to SM-2 quality (0-5)
            anki_rating = int(row['rating'])
            sm2_quality = RatingConverter.anki_to_sm2(anki_rating)
            
            # Handle elapsed_days
            if row['elapsed_days'] >= 0:
                current_time += row['elapsed_days']
            else:
                # First review
                current_time = 0.0
            
            # Process with hybrid scheduler
            try:
                hybrid_decision = hybrid.schedule_review(
                    user_id=user_id,
                    item_id=item_id,
                    rating=sm2_quality,
                    timestamp=current_time
                )
                
                # Process with SM-2 for comparison
                sm2_decision = sm2.schedule_review(
                    user_id=user_id,
                    item_id=item_id,
                    rating=sm2_quality,
                    timestamp=current_time
                )
                
                # Record results
                hybrid_results.append({
                    'card_id': card_id,
                    'review_th': row['review_th'],
                    'rating': anki_rating,
                    'hybrid_interval': hybrid_decision.interval,
                    'hybrid_confidence': hybrid_decision.confidence,
                    'hybrid_p_recall': hybrid_decision.p_recall,
                    'sm2_interval': sm2_decision.interval,
                    'sm2_p_recall': sm2_decision.p_recall,
                    'elapsed_days': row['elapsed_days']
                })
                
                # Show progress for first few cards
                if card_id < 3 and row['review_th'] <= 5:
                    print(f"\nCard {card_id}, Review {row['review_th']}:")
                    print(f"  Rating: {anki_rating} (SM-2: {sm2_quality})")
                    print(f"  Hybrid: interval={hybrid_decision.interval}d, confidence={hybrid_decision.confidence:.2%}, p_recall={hybrid_decision.p_recall:.2%}")
                    print(f"  SM-2:   interval={sm2_decision.interval}d, p_recall={sm2_decision.p_recall:.2%}")
                
            except Exception as e:
                print(f"Error processing card {card_id}, review {row['review_th']}: {e}")
                continue
    
    # Analyze results
    results_df = pd.DataFrame(hybrid_results)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total reviews processed: {len(results_df)}")
    print(f"Cards processed: {len(card_ids)}")
    
    if len(results_df) > 0:
        print(f"\nHybrid Scheduler Statistics:")
        print(f"  Average interval: {results_df['hybrid_interval'].mean():.2f} days")
        print(f"  Average confidence (ML weight): {results_df['hybrid_confidence'].mean():.2%}")
        print(f"  Reviews with ML contribution (>0% confidence): {(results_df['hybrid_confidence'] > 0).sum()} ({(results_df['hybrid_confidence'] > 0).sum() / len(results_df) * 100:.1f}%)")
        print(f"  Average p_recall: {results_df['hybrid_p_recall'].mean():.2%}")
        
        print(f"\nSM-2 Scheduler Statistics:")
        print(f"  Average interval: {results_df['sm2_interval'].mean():.2f} days")
        print(f"  Average p_recall: {results_df['sm2_p_recall'].mean():.2%}")
        
        print(f"\nComparison:")
        avg_interval_diff = results_df['hybrid_interval'].mean() - results_df['sm2_interval'].mean()
        print(f"  Average interval difference: {avg_interval_diff:+.2f} days")
        
        # Show confidence distribution by review number
        print(f"\nML Confidence by Review Number:")
        for review_th in sorted(results_df['review_th'].unique())[:10]:
            subset = results_df[results_df['review_th'] == review_th]
            if len(subset) > 0:
                avg_conf = subset['hybrid_confidence'].mean()
                count = len(subset)
                print(f"  Review {review_th}: {avg_conf:.2%} confidence (n={count})")
    
    return results_df

if __name__ == "__main__":
    # Test with first part of dataset
    results = test_hybrid_with_dataset("fsrs_anki_20k_part_1.csv", max_cards=50)
    
    # Save results
    if len(results) > 0:
        output_file = "hybrid_test_results.csv"
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")



