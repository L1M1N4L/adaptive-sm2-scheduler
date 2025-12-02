"""
Complete research pipeline using SSP-MMC-Plus methodology
1. Preprocess data
2. Train models
3. Evaluate all schedulers
4. Generate visualizations
"""

import sys
import os
import subprocess
import shutil
import pandas as pd
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    eval_dir = project_root / "evaluation" / "SSP-MMC-Plus"
    training_data = eval_dir / "tmp" / "training_data.tsv"
    preprocessed_data = eval_dir / "tmp" / "training_data_preprocessed.tsv"
    models_dir = project_root / "research_output" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COMPLETE RESEARCH PIPELINE - SSP-MMC-Plus METHODOLOGY")
    print("=" * 80)
    
    original_dir = os.getcwd()
    
    try:
        # Step 1: Preprocess data
        print("\n[1/4] Preprocessing data for DHP...")
        from scripts.preprocess_for_dhp import preprocess_for_dhp
        preprocess_for_dhp(str(training_data), str(preprocessed_data))
        
        # Step 2: Train HLR
        print("\n[2/4] Training HLR model...")
        os.chdir(eval_dir)
        result = subprocess.run([
            sys.executable,
            "train.py",
            "-m", "HLR",
            "-l",
            "-train",
            str(preprocessed_data)
        ], capture_output=True, text=True, cwd=str(eval_dir))
        
        if result.returncode == 0:
            print("HLR training completed")
        else:
            print(f"HLR training output: {result.stdout}")
            if result.stderr:
                print(f"HLR training errors: {result.stderr}")
        
        # Find and copy HLR weights
        hlr_weights_src = eval_dir / "tmp" / "HLR_weights.tsv"
        if not hlr_weights_src.exists():
            # Check other locations
            for loc in [eval_dir / "HLR_weights.tsv", eval_dir / "weights.tsv"]:
                if loc.exists():
                    hlr_weights_src = loc
                    break
        
        if hlr_weights_src.exists():
            hlr_weights_dest = models_dir / "HLR_weights.tsv"
            shutil.copy2(hlr_weights_src, hlr_weights_dest)
            print(f"HLR weights saved to: {hlr_weights_dest}")
        else:
            print("Warning: HLR weights not found")
            hlr_weights_dest = None
        
        # Step 3: Train DHP
        print("\n[3/4] Training DHP model...")
        result = subprocess.run([
            sys.executable,
            "train.py",
            "-m", "DHP",
            "-train",
            str(preprocessed_data)
        ], capture_output=True, text=True, cwd=str(eval_dir))
        
        if result.returncode == 0:
            print("DHP training completed")
        else:
            print(f"DHP training output: {result.stdout}")
            if result.stderr:
                print(f"DHP training errors: {result.stderr}")
        
        # Find and copy DHP model
        dhp_model_src = eval_dir / "tmp" / "DHP" / "model.csv"
        if dhp_model_src.exists():
            dhp_model_dest_dir = models_dir / "DHP"
            dhp_model_dest_dir.mkdir(parents=True, exist_ok=True)
            dhp_model_dest = dhp_model_dest_dir / "model.csv"
            shutil.copy2(dhp_model_src, dhp_model_dest)
            print(f"DHP model saved to: {dhp_model_dest}")
        else:
            print("Warning: DHP model not found")
            dhp_model_dest = None
        
        # Step 4: Evaluate using SSP-MMC-Plus methodology
        print("\n[4/4] Evaluating schedulers (SSP-MMC-Plus style)...")
        os.chdir(project_root)
        
        from scripts.evaluate_ssp_mmc_style import evaluate_all_schedulers_ssp_mmc
        
        evaluate_all_schedulers_ssp_mmc(
            str(preprocessed_data),
            hlr_weights_path=str(hlr_weights_dest) if hlr_weights_dest else None,
            dhp_params_path=str(dhp_model_dest) if dhp_model_dest else None,
            random_seed=2022
        )
        
        # Step 5: Generate visualizations from results
        print("\n[5/5] Generating visualizations...")
        from scripts.visualize_hybrid_metrics import main as viz_main
        import sys as sys_module
        
        # Convert SSP-MMC-Plus results to comparison format
        result_dir = Path("./result")
        if result_dir.exists():
            # Combine results from all folds
            comparison_results = []
            for scheduler_name in ["SM-2", "Hybrid", "Pure-ML"]:
                scheduler_dir = result_dir / scheduler_name
                if scheduler_dir.exists():
                    # Load all fold results
                    for result_file in scheduler_dir.glob("repeat*_fold*.tsv"):
                        df = pd.read_csv(result_file, sep='\t')
                        df['scheduler'] = scheduler_name
                        comparison_results.append(df)
            
            if comparison_results:
                comparison_df = pd.concat(comparison_results, ignore_index=True)
                # Convert to our format
                comparison_df['item_id'] = comparison_df.index
                comparison_df['review_num'] = 1
                comparison_df['interval'] = comparison_df['t']
                comparison_df['halflife'] = comparison_df['hh']
                comparison_df['p_recall_pred'] = comparison_df['pp']
                comparison_df['p_recall_actual'] = comparison_df['p']
                comparison_df['ml_confidence'] = 0.0
                comparison_df['ease_factor'] = 2.5
                comparison_df['repetitions'] = 1
                
                output_path = project_root / "research_output" / "comparison_results.csv"
                comparison_df.to_csv(output_path, index=False)
                
                # Generate visualizations
                sys_module.argv = [
                    'visualize_hybrid_metrics.py',
                    str(output_path),
                    '--output-dir',
                    str(project_root / "research_output" / "visualizations")
                ]
                viz_main()
        
        print("\n" + "=" * 80)
        print("RESEARCH PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"Models: {models_dir}")
        print(f"Results: ./result/")
        print(f"Visualizations: research_output/visualizations/")
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()

