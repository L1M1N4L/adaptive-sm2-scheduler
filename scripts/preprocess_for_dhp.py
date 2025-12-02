"""
Preprocess data for DHP training using exact SSP-MMC-Plus methodology
This replicates preprocess.py exactly
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

d2p = [0.86, 0.78, 0.72, 0.66, 0.61, 0.55, 0.49, 0.44, 0.39, 0.34]

def cal_halflife(group):
    """Calculate halflife for a group - exact copy from preprocess.py"""
    if group['i'].values[0] > 1:
        r_ivl_cnt = sum(group['delta_t'] * group['p_recall'].map(np.log) * group['total_cnt'])
        ivl_ivl_cnt = sum(group['delta_t'].map(lambda x: x ** 2) * group['total_cnt'])
        group['halflife'] = round(np.log(0.5) / (r_ivl_cnt / ivl_ivl_cnt), 4)
    else:
        group['halflife'] = 0.0
    group['group_cnt'] = sum(group['total_cnt'])
    return group

def preprocess_for_dhp(input_path: str, output_path: str):
    """
    Preprocess data exactly as SSP-MMC-Plus preprocess.py does.
    """
    print("Preprocessing data for DHP (SSP-MMC-Plus style)...")
    
    # Step 1: Load and filter (exact same as preprocess.py line 33-34)
    data = pd.read_csv(input_path, sep='\t', index_col=None)
    data = data[(data['p_recall'] < 1) & (data['p_recall'] > 0)]
    
    # Step 2: Calculate halflife by grouping (exact same as preprocess.py line 35-37)
    print("Calculating halflife by groups...")
    grouped = data.groupby(by=['d', 'i', 'r_history', 't_history'])
    data = grouped.apply(cal_halflife)
    
    # Reset index after groupby - drop index levels that are already columns
    if isinstance(data.index, pd.MultiIndex):
        # Get index level names
        index_names = data.index.names
        # Check which are already columns
        to_drop = [name for name in index_names if name in data.columns]
        data = data.reset_index(level=to_drop, drop=True)
        # Reset remaining levels
        if len([n for n in index_names if n not in to_drop]) > 0:
            data = data.reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)
    
    # Step 3: Round p_recall (exact same as preprocess.py line 39)
    data['p_recall'] = data['p_recall'].map(lambda x: round(x, 2))
    
    # Step 4: Initialize p_history (exact same as preprocess.py line 40-41)
    data['p_history'] = '0'
    data.sort_values('i', inplace=True)
    
    # Step 5: Set p_history for i=2 (exact same as preprocess.py line 45-46)
    print("Setting initial p_history...")
    for idx in tqdm(data[(data['i'] == 2)].index):
        data.loc[idx, 'p_history'] = d2p[data.loc[idx, 'd']-1]
    
    data['p_history'] = data['p_history'].map(lambda x: str(x))
    
    # Step 6: Add last_halflife and last_p_recall (exact same as preprocess.py line 50-58)
    print("Adding last_halflife and last_p_recall...")
    # Initialize columns
    data['last_halflife'] = np.nan
    data['last_p_recall'] = np.nan
    
    for idx in tqdm(data[data['i'] >= 2].index):
        item = data.loc[idx]
        interval = int(item['delta_t'])
        
        # Find the next review that this one leads to
        # Match: r_history starts with current, t_history = current + interval, same d
        # This is exactly what preprocess.py does (line 53-55)
        index = data[(data['r_history'].str.startswith(item['r_history'])) & (
                data['t_history'] == item['t_history'] + f',{interval}') & (
                             data['d'] == item['d'])].index
        
        if len(index) > 0:
            # Set last_halflife and last_p_recall for the NEXT review (line 56-58)
            data.loc[index, 'p_history'] = item['p_history'] + ',' + str(item['p_recall'])
            data.loc[index, 'last_halflife'] = item['halflife']
            data.loc[index, 'last_p_recall'] = item['p_recall']
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, sep='\t', index=False)
    print(f"Preprocessed data saved to: {output_path}")
    print(f"Rows with last_halflife: {data['last_halflife'].notna().sum()} / {len(data)}")
    return str(output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for DHP training')
    parser.add_argument('input_tsv', help='Input TSV file')
    parser.add_argument('output_tsv', help='Output TSV file')
    
    args = parser.parse_args()
    preprocess_for_dhp(args.input_tsv, args.output_tsv)

