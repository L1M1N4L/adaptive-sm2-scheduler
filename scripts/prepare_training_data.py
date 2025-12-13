"""
Convert FSRS-Anki dataset to SSP-MMC-Plus training format
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm

def convert_fsrs_to_ssp_mmc(fsrs_csv_path: str, output_tsv_path: str, max_cards: int = None):
    """
    Convert FSRS-Anki dataset to SSP-MMC-Plus training format.
    
    Expected output format (TSV):
    - halflife: memory half-life
    - i: review number
    - p_recall: predicted recall probability
    - delta_t: time since last review
    - r_history: comma-separated recall history (1=correct, 0=wrong)
    - t_history: comma-separated time intervals
    - p_history: comma-separated predicted recall probabilities
    - d: difficulty (1-10)
    - total_cnt: total review count
    """
    print(f"Loading FSRS dataset from {fsrs_csv_path}...")
    df = pd.read_csv(fsrs_csv_path)
    
    print(f"Dataset loaded: {len(df)} rows")
    
    # Group by card_id
    card_ids = df['card_id'].unique()
    if max_cards:
        card_ids = card_ids[:max_cards]
    
    print(f"Processing {len(card_ids)} cards...")
    
    training_data = []
    
    for card_id in tqdm(card_ids):
        card_data = df[df['card_id'] == card_id].sort_values('review_th')
        
        if len(card_data) < 2:  # Need at least 2 reviews
            continue
        
        # Build review history
        r_history = []
        t_history = []
        p_history = []
        last_timestamp = 0.0
        current_time = 0.0
        
        # Calculate difficulty from early reviews (wrong answers)
        wrong_count = sum(1 for _, row in card_data.iterrows() if row['rating'] < 3)
        difficulty = min(max(1, wrong_count + 1), 10)  # 1-10 scale
        
        for idx, row in card_data.iterrows():
            # Convert Anki rating (1-4) to recall (1=correct, 0=wrong)
            # Rating: 1=Again, 2=Hard, 3=Good, 4=Easy
            # We'll treat 3-4 as correct (1), 1-2 as wrong (0)
            recall = 1 if row['rating'] >= 3 else 0
            
            # Calculate elapsed time
            if row['elapsed_days'] >= 0:
                elapsed_days = row['elapsed_days']
            else:
                elapsed_days = 0  # First review
            
            # Update current time
            if elapsed_days > 0:
                current_time += elapsed_days
            elif idx == card_data.index[0]:
                current_time = 0.0
            
            # Calculate interval since last review
            if len(t_history) > 0:
                interval = max(1, int(elapsed_days)) if elapsed_days > 0 else 1
            else:
                interval = 0  # First review
            
            # Estimate half-life and recall probability
            # For first review, use default
            if len(r_history) == 0:
                halflife = 1.0
                p_recall = 0.5
            else:
                # Estimate half-life from previous interval and recall
                prev_interval = int(t_history[-1]) if t_history[-1] != '0' else 1
                prev_recall = int(r_history[-1])
                
                if prev_recall == 1:
                    # Correct recall - half-life increases
                    halflife = max(1.0, prev_interval * 1.5)
                else:
                    # Wrong recall - half-life decreases
                    halflife = max(0.5, prev_interval * 0.5)
                
                # Calculate predicted recall
                p_recall = 2.0 ** (-interval / halflife) if halflife > 0 else 0.1
            
            # Add to history
            r_history.append(str(recall))
            t_history.append(str(interval))
            p_history.append(f"{p_recall:.4f}")
            
            # Create training sample (skip first review as it has no history)
            if len(r_history) >= 2:
                training_data.append({
                    'halflife': halflife,
                    'i': len(r_history),
                    'p_recall': p_recall,
                    'delta_t': interval,
                    'r_history': ','.join(r_history[:-1]),  # Exclude current
                    't_history': ','.join(t_history[:-1]),  # Exclude current
                    'p_history': ','.join(p_history[:-1]),  # Exclude current
                    'd': difficulty,
                    'total_cnt': len(r_history) - 1
                })
    
    # Create DataFrame
    train_df = pd.DataFrame(training_data)
    
    # Filter valid data
    train_df = train_df[train_df['halflife'] > 0]
    train_df = train_df[train_df['i'] > 0]
    
    # Ensure history lengths match
    def check_history_lengths(row):
        r_len = len(str(row['r_history']).split(','))
        t_len = len(str(row['t_history']).split(','))
        p_len = len(str(row['p_history']).split(','))
        return r_len == t_len == p_len
    
    train_df = train_df[train_df.apply(check_history_lengths, axis=1)]
    
    print(f"\nConverted {len(train_df)} training samples")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"\nSample data:")
    print(train_df.head())
    
    # Save as TSV
    output_path = Path(output_tsv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved training data to {output_path}")
    
    return train_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert FSRS-Anki dataset to SSP-MMC-Plus format')
    parser.add_argument('input_csv', help='Input FSRS-Anki CSV file')
    parser.add_argument('output_tsv', help='Output TSV file for training')
    parser.add_argument('--max-cards', type=int, default=None, help='Maximum number of cards to process')
    
    args = parser.parse_args()
    
    convert_fsrs_to_ssp_mmc(args.input_csv, args.output_tsv, args.max_cards)



