"""
Full pipeline: Convert data -> Train models -> Simulate with hybrid scheduler
"""

import subprocess
import sys
from pathlib import Path

def run_pipeline(
    fsrs_csv: str,
    output_dir: str = "./evaluation/SSP-MMC-Plus/tmp",
    max_cards: int = 5000
):
    """Run the complete pipeline."""
    
    print("=" * 60)
    print("HYBRID SCHEDULER TRAINING & SIMULATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Convert data
    print("\n[Step 1/3] Converting FSRS-Anki data to SSP-MMC-Plus format...")
    training_data_path = Path(output_dir) / "training_data.tsv"
    training_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    subprocess.run([
        sys.executable,
        "scripts/prepare_training_data.py",
        fsrs_csv,
        str(training_data_path),
        "--max-cards", str(max_cards)
    ], check=True)
    
    # Step 2: Train models
    print("\n[Step 2/3] Training HLR and DHP models...")
    subprocess.run([
        sys.executable,
        "scripts/train_hybrid_models.py",
        str(training_data_path),
        "--output-dir", output_dir
    ], check=True)
    
    # Step 3: Simulate
    print("\n[Step 3/3] Running simulation with hybrid scheduler...")
    hlr_weights = Path(output_dir) / "HLR_weights.tsv"
    dhp_params = Path(output_dir) / "DHP" / "model.csv"
    
    subprocess.run([
        sys.executable,
        "scripts/simulate_hybrid.py",
        str(training_data_path),
        "--hlr-weights", str(hlr_weights) if hlr_weights.exists() else "",
        "--dhp-params", str(dhp_params) if dhp_params.exists() else "",
        "--max-cards", str(max_cards),
        "--output", "./simulation/hybrid_results.tsv"
    ], check=True)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  - Training data: {training_data_path}")
    print(f"  - HLR weights: {hlr_weights}")
    print(f"  - DHP params: {dhp_params}")
    print(f"  - Simulation results: ./simulation/hybrid_results.tsv")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full training and simulation pipeline')
    parser.add_argument('fsrs_csv', help='Input FSRS-Anki CSV file')
    parser.add_argument('--output-dir', default='./evaluation/SSP-MMC-Plus/tmp', help='Output directory')
    parser.add_argument('--max-cards', type=int, default=5000, help='Maximum cards to process')
    
    args = parser.parse_args()
    
    run_pipeline(args.fsrs_csv, args.output_dir, args.max_cards)



