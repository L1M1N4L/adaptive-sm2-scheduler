"""
Train all models (HLR, DHP, SM-2 baseline) for comparison
"""

import sys
import os
import subprocess
from pathlib import Path

def train_all_models(data_tsv: str, output_dir: str = "./evaluation/SSP-MMC-Plus/tmp"):
    """Train all models for comparison."""
    
    print("=" * 60)
    print("TRAINING ALL MODELS FOR COMPARISON")
    print("=" * 60)
    
    eval_dir = Path(__file__).parent.parent / "evaluation" / "SSP-MMC-Plus"
    original_dir = os.getcwd()
    
    try:
        os.chdir(eval_dir)
        
        # Train HLR
        print("\n[1/2] Training HLR model...")
        subprocess.run([
            sys.executable,
            "train.py",
            "-m", "HLR",
            "-l",  # Omit lexemes
            "-train",
            data_tsv
        ], check=True, cwd=str(eval_dir))
        
        # Train DHP
        print("\n[2/2] Training DHP model...")
        subprocess.run([
            sys.executable,
            "train.py",
            "-m", "DHP",
            "-train",
            data_tsv
        ], check=True, cwd=str(eval_dir))
        
        print("\n" + "=" * 60)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 60)
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all models for comparison')
    parser.add_argument('data_tsv', help='Training data TSV file')
    parser.add_argument('--output-dir', default='./evaluation/SSP-MMC-Plus/tmp', help='Output directory')
    
    args = parser.parse_args()
    train_all_models(args.data_tsv, args.output_dir)



