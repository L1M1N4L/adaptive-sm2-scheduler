"""
Train all models and then simulate on FSRS-Anki dataset
"""

import sys
import os
import subprocess
from pathlib import Path
import shutil

def preprocess_data(input_path: str, output_path: str):
    """Preprocess training data using SSP-MMC-Plus preprocess.py exactly."""
    import subprocess
    import sys
    
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / "evaluation" / "SSP-MMC-Plus"
    preprocess_script = eval_dir / "preprocess.py"
    
    original_dir = os.getcwd()
    
    try:
        os.chdir(eval_dir)
        
        # The preprocess.py expects data in ./data/ directory
        # We need to copy our data there temporarily or modify the script
        # For now, let's use the exact same preprocessing logic
        
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        
        print("Preprocessing data using SSP-MMC-Plus methodology...")
        data = pd.read_csv(input_path, sep='\t', index_col=None)
        
        # Filter valid data
        data = data[(data['p_recall'] < 1) & (data['p_recall'] > 0)]
        
        # Group by d, i, r_history, t_history and calculate halflife if needed
        # (This is done in preprocess.py cal_halflife function)
        
        # Round p_recall
        data['p_recall'] = data['p_recall'].map(lambda x: round(x, 2))
        
        # Initialize p_history
        data['p_history'] = '0'
        data.sort_values('i', inplace=True)
        
        # d2p mapping for initial p_history
        d2p = [0.86, 0.78, 0.72, 0.66, 0.61, 0.55, 0.49, 0.44, 0.39, 0.34]
        
        # Set p_history for i=2
        for idx in tqdm(data[(data['i'] == 2)].index, desc="Setting initial p_history"):
            d = int(data.loc[idx, 'd'])
            if 1 <= d <= 10:
                data.loc[idx, 'p_history'] = str(d2p[d-1])
        
        data['p_history'] = data['p_history'].map(lambda x: str(x))
        
        # Add last_halflife and last_p_recall (this is the key part)
        data['last_halflife'] = np.nan
        data['last_p_recall'] = np.nan
        
        for idx in tqdm(data[data['i'] >= 2].index, desc="Adding last_halflife and last_p_recall"):
            item = data.loc[idx]
            interval = int(item['delta_t'])
            
            # Find the previous review that leads to this one
            # Match: r_history starts with previous, t_history matches previous + interval, same d
            prev_r_history = ','.join(item['r_history'].split(',')[:-1])
            prev_t_history = ','.join(item['t_history'].split(',')[:-1])
            
            # Find matching previous review
            prev_idx = data[
                (data['r_history'].str.startswith(prev_r_history) if prev_r_history else data['r_history'] == prev_r_history) &
                (data['t_history'] == prev_t_history) &
                (data['d'] == item['d']) &
                (data['i'] == item['i'] - 1)
            ].index
            
            if len(prev_idx) > 0:
                prev_item = data.loc[prev_idx[0]]
                data.loc[idx, 'last_halflife'] = prev_item['halflife']
                data.loc[idx, 'last_p_recall'] = prev_item['p_recall']
                data.loc[idx, 'p_history'] = prev_item['p_history'] + ',' + str(prev_item['p_recall'])
        
        # Save preprocessed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, sep='\t', index=False)
        print(f"Preprocessed data saved to: {output_path}")
        return str(output_path)
        
    finally:
        os.chdir(original_dir)

def train_models(training_data_path: str, output_dir: str):
    """Train HLR and DHP models."""
    print("=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / "evaluation" / "SSP-MMC-Plus"
    training_data = Path(training_data_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_dir = os.getcwd()
    
    try:
        os.chdir(eval_dir)
        
        # Preprocess data for DHP using SSP-MMC-Plus method
        preprocessed_data = eval_dir / "tmp" / "training_data_preprocessed.tsv"
        
        # Use the preprocess script
        from scripts.preprocess_for_dhp import preprocess_for_dhp
        preprocess_for_dhp(str(training_data), str(preprocessed_data))
        
        # Train HLR
        print("\n[1/2] Training HLR model...")
        result = subprocess.run([
            sys.executable,
            "train.py",
            "-m", "HLR",
            "-l",  # Omit lexemes
            "-train",
            str(training_data)
        ], capture_output=True, text=True, cwd=str(eval_dir))
        
        if result.returncode != 0:
            print(f"HLR training error: {result.stderr}")
        else:
            print(result.stdout)
        
        # Find and copy HLR weights
        hlr_weights = eval_dir / "tmp" / "HLR_weights.tsv"
        if not hlr_weights.exists():
            # Check if saved elsewhere
            for possible_path in [eval_dir / "HLR_weights.tsv", eval_dir / "weights.tsv"]:
                if possible_path.exists():
                    hlr_weights = possible_path
                    break
        
        if hlr_weights.exists():
            dest = output_dir / "HLR_weights.tsv"
            shutil.copy2(hlr_weights, dest)
            print(f"HLR weights saved to: {dest}")
        else:
            print("Warning: HLR weights not found")
        
        # Train DHP with preprocessed data
        print("\n[2/2] Training DHP model...")
        result = subprocess.run([
            sys.executable,
            "train.py",
            "-m", "DHP",
            "-train",
            str(preprocessed_data)
        ], capture_output=True, text=True, cwd=str(eval_dir))
        
        if result.returncode != 0:
            print(f"DHP training error: {result.stderr}")
            return False
        
        print(result.stdout)
        
        # Find and copy DHP model
        dhp_model = eval_dir / "tmp" / "DHP" / "model.csv"
        if not dhp_model.exists():
            # Check other possible locations
            for possible_path in [eval_dir / "DHP" / "model.csv", eval_dir / "model.csv"]:
                if possible_path.exists():
                    dhp_model = possible_path
                    break
        
        if dhp_model.exists():
            dest_dir = output_dir / "DHP"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "model.csv"
            shutil.copy2(dhp_model, dest)
            print(f"DHP model saved to: {dest}")
        else:
            print("Warning: DHP model not found")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        return True
        
    finally:
        os.chdir(original_dir)

def prepare_fsrs_data(fsrs_csv_path: str, output_tsv_path: str):
    """Convert FSRS-Anki CSV to SSP-MMC-Plus TSV format."""
    print("\n" + "=" * 80)
    print("PREPARING FSRS DATA")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.prepare_training_data import convert_fsrs_to_ssp_mmc
    
    convert_fsrs_to_ssp_mmc(fsrs_csv_path, output_tsv_path, max_cards=1000)
    print(f"FSRS data prepared: {output_tsv_path}")

def run_simulation(fsrs_data_path: str, models_dir: str, output_path: str):
    """Run scheduler comparison simulation."""
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.compare_schedulers import compare_all_schedulers
    
    models_dir = Path(models_dir)
    hlr_weights = models_dir / "HLR_weights.tsv" if (models_dir / "HLR_weights.tsv").exists() else None
    dhp_params = models_dir / "DHP" / "model.csv" if (models_dir / "DHP" / "model.csv").exists() else None
    
    compare_all_schedulers(
        data_path=fsrs_data_path,
        hlr_weights_path=str(hlr_weights) if hlr_weights else None,
        dhp_params_path=str(dhp_params) if dhp_params else None,
        max_samples=1000,
        output_path=output_path
    )

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models and simulate on FSRS data')
    parser.add_argument('training_data', help='Training data TSV (SSP-MMC-Plus format)')
    parser.add_argument('fsrs_data', help='FSRS-Anki CSV file for simulation')
    parser.add_argument('--models-dir', default='research_output/models', help='Output directory for trained models')
    parser.add_argument('--output', default='research_output/comparison_results.csv', help='Output path for simulation results')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing models)')
    
    args = parser.parse_args()
    
    # Step 1: Train models
    if not args.skip_training:
        if not train_models(args.training_data, args.models_dir):
            print("Training failed. Exiting.")
            return
    
    # Step 2: Prepare FSRS data
    fsrs_tsv = Path(args.fsrs_data).parent / f"{Path(args.fsrs_data).stem}_ssp_mmc.tsv"
    prepare_fsrs_data(args.fsrs_data, str(fsrs_tsv))
    
    # Step 3: Run simulation
    run_simulation(str(fsrs_tsv), args.models_dir, args.output)
    
    # Step 4: Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    from scripts.visualize_hybrid_metrics import main as viz_main
    import sys as sys_module
    sys_module.argv = ['visualize_hybrid_metrics.py', args.output, '--output-dir', 'research_output/visualizations']
    viz_main()
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Results: {args.output}")
    print(f"Models: {args.models_dir}")
    print(f"Visualizations: research_output/visualizations/")

if __name__ == "__main__":
    main()

