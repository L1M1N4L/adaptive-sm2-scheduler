"""
Complete research pipeline: Train -> Compare -> Visualize
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_research_pipeline(
    fsrs_csv: str,
    max_cards: int = 5000,
    max_samples: int = 1000,
    output_base: str = "./research_output"
):
    """Run complete research pipeline."""
    
    print("=" * 70)
    print("RESEARCH PIPELINE: TRAIN -> COMPARE -> VISUALIZE")
    print("=" * 70)
    
    base_path = Path(output_base)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare data
    print("\n" + "=" * 70)
    print("[STEP 1/5] Preparing training data...")
    print("=" * 70)
    training_data = base_path / "training_data.tsv"
    subprocess.run([
        sys.executable,
        "scripts/prepare_training_data.py",
        fsrs_csv,
        str(training_data),
        "--max-cards", str(max_cards)
    ], check=True)
    
    # Step 2: Train models
    print("\n" + "=" * 70)
    print("[STEP 2/5] Training models...")
    print("=" * 70)
    model_dir = base_path / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Train HLR
    print("\nTraining HLR model...")
    subprocess.run([
        sys.executable,
        "scripts/train_hybrid_models.py",
        str(training_data),
        "--output-dir", str(model_dir),
        "--hlr-only"
    ], check=True)
    
    # Train DHP
    print("\nTraining DHP model...")
    subprocess.run([
        sys.executable,
        "scripts/train_hybrid_models.py",
        str(training_data),
        "--output-dir", str(model_dir),
        "--dhp-only"
    ], check=True)
    
    # Step 3: Compare schedulers
    print("\n" + "=" * 70)
    print("[STEP 3/5] Comparing schedulers...")
    print("=" * 70)
    hlr_weights = model_dir / "HLR_weights.tsv"
    dhp_params = model_dir / "DHP" / "model.csv"
    
    comparison_results = base_path / "comparison_results.csv"
    comparison_metrics = base_path / "comparison_metrics.csv"
    
    subprocess.run([
        sys.executable,
        "scripts/compare_schedulers.py",
        str(training_data),
        "--hlr-weights", str(hlr_weights) if hlr_weights.exists() else "",
        "--dhp-params", str(dhp_params) if dhp_params.exists() else "",
        "--max-samples", str(max_samples),
        "--output", str(comparison_results)
    ], check=True)
    
    # Step 4: Create visualizations
    print("\n" + "=" * 70)
    print("[STEP 4/5] Creating visualizations...")
    print("=" * 70)
    viz_dir = base_path / "visualizations"
    subprocess.run([
        sys.executable,
        "scripts/visualize_comparison.py",
        str(comparison_results),
        str(comparison_metrics),
        "--output-dir", str(viz_dir)
    ], check=True)
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("[STEP 5/5] Generating summary...")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("RESEARCH PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {base_path}")
    print("\nGenerated files:")
    print(f"  - Training data: {training_data}")
    print(f"  - HLR weights: {hlr_weights}")
    print(f"  - DHP parameters: {dhp_params}")
    print(f"  - Comparison results: {comparison_results}")
    print(f"  - Comparison metrics: {comparison_metrics}")
    print(f"  - Visualizations: {viz_dir}/")
    for viz_file in viz_dir.glob("*.png"):
        print(f"      - {viz_file.name}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review visualizations in:", viz_dir)
    print("2. Analyze metrics in:", comparison_metrics)
    print("3. Use trained models in your hybrid scheduler")
    print("4. Write up research findings")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run complete research pipeline')
    parser.add_argument('fsrs_csv', help='Input FSRS-Anki CSV file')
    parser.add_argument('--max-cards', type=int, default=5000, help='Max cards for training')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples for comparison')
    parser.add_argument('--output', default='./research_output', help='Output directory')
    
    args = parser.parse_args()
    
    run_research_pipeline(
        args.fsrs_csv,
        max_cards=args.max_cards,
        max_samples=args.max_samples,
        output_base=args.output
    )



