"""
Complete research output generation script.
Compares SM-2, Hybrid, and Pure ML schedulers and generates all visualizations.
"""

import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("COMPLETE RESEARCH OUTPUT GENERATION")
    print("=" * 80)
    
    # Step 1: Run comparison with all three schedulers
    print("\n[1/3] Running scheduler comparison (SM-2, Hybrid, Pure ML)...")
    compare_cmd = [
        sys.executable,
        str(project_root / "scripts" / "compare_schedulers.py"),
        str(project_root / "research_output" / "training_data.tsv"),
        "--hlr-weights", str(project_root / "research_output" / "models" / "HLR_weights.tsv"),
        "--dhp-params", str(project_root / "research_output" / "models" / "DHP" / "model.csv"),
        "--max-samples", "1000",
        "--output", str(project_root / "research_output" / "comparison_results.csv")
    ]
    
    result = subprocess.run(compare_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running comparison: {result.stderr}")
        return
    print(result.stdout)
    
    # Step 2: Generate visualizations
    print("\n[2/3] Generating visualizations...")
    viz_cmd = [
        sys.executable,
        str(project_root / "scripts" / "visualize_hybrid_metrics.py"),
        str(project_root / "research_output" / "comparison_results.csv"),
        "--output-dir", str(project_root / "research_output" / "visualizations")
    ]
    
    result = subprocess.run(viz_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating visualizations: {result.stderr}")
        return
    print(result.stdout)
    
    # Step 3: Generate research summary report
    print("\n[3/3] Generating research summary report...")
    generate_report(project_root)
    
    print("\n" + "=" * 80)
    print("RESEARCH OUTPUT GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {project_root / 'research_output'}")
    print("\nGenerated files:")
    print("  - comparison_results.csv")
    print("  - comparison_metrics.csv")
    print("  - visualizations/recall_curves.png")
    print("  - visualizations/thr_metrics.png")
    print("  - visualizations/srp_metrics.png")
    print("  - visualizations/wtl_metrics.png")
    print("  - visualizations/daily_cost.png")
    print("  - visualizations/efficiency.png")
    print("  - visualizations/comprehensive_metrics.png")
    print("  - RESEARCH_REPORT.md")


def generate_report(project_root):
    """Generate comprehensive research report."""
    import pandas as pd
    import numpy as np
    
    # Load metrics
    metrics_df = pd.read_csv(project_root / "research_output" / "comparison_metrics.csv")
    results_df = pd.read_csv(project_root / "research_output" / "comparison_results.csv")
    
    # Calculate additional metrics
    report_lines = []
    report_lines.append("# Research Report: Hybrid SM-2 + ML Scheduler")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This report compares three spaced repetition scheduling approaches:")
    report_lines.append("1. **SM-2**: Traditional SuperMemo algorithm (baseline)")
    report_lines.append("2. **Hybrid**: SM-2 + ML adaptive blending")
    report_lines.append("3. **Pure ML**: ML-only predictions (no SM-2 blending)")
    report_lines.append("")
    
    report_lines.append("## Key Metrics Comparison")
    report_lines.append("")
    report_lines.append("| Metric | SM-2 | Hybrid | Pure ML |")
    report_lines.append("|--------|------|--------|---------|")
    
    for _, row in metrics_df.iterrows():
        scheduler = row['scheduler']
        if scheduler not in ['SM-2', 'Hybrid', 'Pure-ML']:
            continue
        
        mae = row['mae_p']
        avg_interval = row['avg_interval']
        avg_halflife = row['avg_halflife']
        avg_recall = row['avg_p_recall']
        ml_conf = row['avg_ml_confidence']
        
        if scheduler == 'SM-2':
            sm2_mae = mae
            sm2_interval = avg_interval
            sm2_halflife = avg_halflife
            sm2_recall = avg_recall
        elif scheduler == 'Hybrid':
            hybrid_mae = mae
            hybrid_interval = avg_interval
            hybrid_halflife = avg_halflife
            hybrid_recall = avg_recall
        elif scheduler == 'Pure-ML':
            ml_mae = mae
            ml_interval = avg_interval
            ml_halflife = avg_halflife
            ml_recall = avg_recall
    
    # Add detailed metrics
    report_lines.append("### Prediction Accuracy (MAE)")
    report_lines.append("")
    for _, row in metrics_df.iterrows():
        scheduler = row['scheduler']
        if scheduler not in ['SM-2', 'Hybrid', 'Pure-ML']:
            continue
        mae = row['mae_p']
        report_lines.append(f"- **{scheduler}**: {mae:.4f}")
    report_lines.append("")
    
    report_lines.append("### Average Intervals")
    report_lines.append("")
    for _, row in metrics_df.iterrows():
        scheduler = row['scheduler']
        if scheduler not in ['SM-2', 'Hybrid', 'Pure-ML']:
            continue
        interval = row['avg_interval']
        report_lines.append(f"- **{scheduler}**: {interval:.2f} days")
    report_lines.append("")
    
    report_lines.append("### Average Half-life")
    report_lines.append("")
    for _, row in metrics_df.iterrows():
        scheduler = row['scheduler']
        if scheduler not in ['SM-2', 'Hybrid', 'Pure-ML']:
            continue
        halflife = row['avg_halflife']
        report_lines.append(f"- **{scheduler}**: {halflife:.2f} days")
    report_lines.append("")
    
    # Calculate recall curves at key days
    report_lines.append("## Recall Curves at Key Time Points")
    report_lines.append("")
    report_lines.append("| Days | SM-2 | Hybrid | Pure ML |")
    report_lines.append("|------|------|--------|---------|")
    
    # Load detailed metrics if available
    detailed_metrics_path = project_root / "research_output" / "visualizations" / "hybrid_metrics_detailed.csv"
    if detailed_metrics_path.exists():
        detailed_df = pd.read_csv(detailed_metrics_path)
        for day in [30, 60, 180, 365]:
            day_data = detailed_df[detailed_df['day'] == day]
            if len(day_data) > 0:
                sm2_recall = day_data[day_data['scheduler'] == 'SM-2']['avg_recall'].values[0] if len(day_data[day_data['scheduler'] == 'SM-2']) > 0 else 0
                hybrid_recall = day_data[day_data['scheduler'] == 'Hybrid']['avg_recall'].values[0] if len(day_data[day_data['scheduler'] == 'Hybrid']) > 0 else 0
                ml_recall = day_data[day_data['scheduler'] == 'Pure-ML']['avg_recall'].values[0] if len(day_data[day_data['scheduler'] == 'Pure-ML']) > 0 else 0
                report_lines.append(f"| {day} | {sm2_recall:.3f} | {hybrid_recall:.3f} | {ml_recall:.3f} |")
    
    report_lines.append("")
    report_lines.append("## Visualizations")
    report_lines.append("")
    report_lines.append("All visualizations are available in `research_output/visualizations/`:")
    report_lines.append("")
    report_lines.append("- `recall_curves.png`: Average recall probability over time")
    report_lines.append("- `thr_metrics.png`: Target Half-life Reached (THR) metrics")
    report_lines.append("- `srp_metrics.png`: Summation of Recall Probability (SRP)")
    report_lines.append("- `wtl_metrics.png`: Words Total Learned (WTL)")
    report_lines.append("- `daily_cost.png`: Daily review cost (reviews/day)")
    report_lines.append("- `efficiency.png`: Efficiency (SRP / Cost)")
    report_lines.append("- `comprehensive_metrics.png`: All metrics in one view")
    report_lines.append("")
    
    report_lines.append("## Conclusions")
    report_lines.append("")
    report_lines.append("(Analysis and conclusions based on the metrics above)")
    report_lines.append("")
    
    # Write report
    report_path = project_root / "research_output" / "RESEARCH_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Research report saved to: {report_path}")


if __name__ == "__main__":
    main()

