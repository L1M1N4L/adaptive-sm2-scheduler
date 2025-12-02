"""
Train HLR and DHP models for hybrid scheduler
"""

import sys
import os
from pathlib import Path

def train_hlr_model(data_path: str, output_dir: str = "./evaluation/SSP-MMC-Plus/tmp"):
    """Train HLR model and save weights."""
    print("=" * 60)
    print("Training HLR Model")
    print("=" * 60)
    
    # Convert to absolute path before changing directory
    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    original_dir = os.getcwd()
    eval_dir = Path(__file__).parent.parent / "evaluation" / "SSP-MMC-Plus"
    
    try:
        os.chdir(eval_dir)
        
        # Import after changing directory
        from train import load_data, feature_extract
        from model.halflife_regression import SpacedRepetitionModel
        
        # Load data (use absolute path)
        dataset = load_data(str(data_path))
        train = dataset.sample(frac=0.8, random_state=2022)
        test = dataset.drop(index=train.index)
        
        train_train, train_test = train.sample(frac=0.5, random_state=2022)
        train_test = train.drop(index=train_train.index)
        
        print(f"Training set: {len(train_train)} samples")
        print(f"Test set: {len(train_test)} samples")
        
        # Extract features
        train_fold, test_fold = feature_extract(train_train, train_test, 'HLR', omit_lexemes=True)
        
        # Train model
        model = SpacedRepetitionModel(train_fold, test_fold, method='HLR', omit_lexemes=True)
        model.train()
        
        # Save weights
        weights_path = output_dir / "HLR_weights.tsv"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model.dump_weights(str(weights_path))
        
        print(f"HLR weights saved to {weights_path}")
        
        # Evaluate
        model.eval(0, 0)
        
        return str(weights_path), model
    finally:
        os.chdir(original_dir)

def train_dhp_model(data_path: str, output_dir: str = "./evaluation/SSP-MMC-Plus/tmp"):
    """Train DHP model and save parameters."""
    print("\n" + "=" * 60)
    print("Training DHP Model")
    print("=" * 60)
    
    # Convert to absolute path before changing directory
    data_path = Path(data_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    original_dir = os.getcwd()
    eval_dir = Path(__file__).parent.parent / "evaluation" / "SSP-MMC-Plus"
    
    try:
        os.chdir(eval_dir)
        
        # Import after changing directory
        from train import load_data
        from model.DHP import SpacedRepetitionModel
        
        # Load data (use absolute path)
        dataset = load_data(str(data_path))
        train = dataset.sample(frac=0.8, random_state=2022)
        test = dataset.drop(index=train.index)
        
        train_train, train_test = train.sample(frac=0.5, random_state=2022)
        train_test = train.drop(index=train_train.index)
        
        print(f"Training set: {len(train_train)} samples")
        print(f"Test set: {len(train_test)} samples")
        
        # Train model
        model = SpacedRepetitionModel(train_train, train_test)
        model.train()
        
        # Save parameters
        params_path = output_dir / "DHP" / "model.csv"
        params_path.parent.mkdir(parents=True, exist_ok=True)
        model.save()
        
        print(f"DHP parameters saved to {params_path}")
        
        # Evaluate
        model.eval(0, 0)
        
        return str(params_path), model
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HLR and DHP models for hybrid scheduler')
    parser.add_argument('data_tsv', help='Training data in SSP-MMC-Plus TSV format')
    parser.add_argument('--output-dir', default='./evaluation/SSP-MMC-Plus/tmp', help='Output directory for models')
    parser.add_argument('--hlr-only', action='store_true', help='Train only HLR model')
    parser.add_argument('--dhp-only', action='store_true', help='Train only DHP model')
    
    args = parser.parse_args()
    
    if not args.dhp_only:
        train_hlr_model(args.data_tsv, args.output_dir)
    
    if not args.hlr_only:
        train_dhp_model(args.data_tsv, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {args.output_dir}")
    print("\nTo use trained models in hybrid scheduler:")
    print(f"  hybrid = HybridScheduler(")
    print(f"      hlr_weights_path='{args.output_dir}/HLR_weights.tsv',")
    print(f"      dhp_params_path='{args.output_dir}/DHP/model.csv'")
    print(f"  )")

