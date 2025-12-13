# Research Output Directory

This directory contains all research outputs comparing SM-2, Hybrid, and Pure ML schedulers.

## Quick Start

### View Results
```bash
# Summary metrics
cat comparison_metrics.csv

# Detailed results
head -20 comparison_results.csv

# Research report
cat RESEARCH_REPORT.md
```

### View Visualizations
All visualizations are in the `visualizations/` folder:
- Open PNG files to view charts
- Windows: `start visualizations\recall_curves.png`
- Mac/Linux: `open visualizations/recall_curves.png`

## File Structure

```
research_output/
├── comparison_results.csv          # Detailed per-review results
├── comparison_metrics.csv          # Summary metrics table
├── RESEARCH_REPORT.md              # Comprehensive research report
├── RESULTS_SUMMARY.md              # Quick reference summary
├── training_data.tsv               # Training dataset
├── models/                         # Trained ML models
│   ├── HLR_weights.tsv
│   └── DHP/model.csv
└── visualizations/                 # All visualization files
    ├── recall_curves.png           # Recall probability over time
    ├── thr_metrics.png             # Target Half-life Reached
    ├── srp_metrics.png             # Summation of Recall Probability
    ├── wtl_metrics.png             # Words Total Learned
    ├── daily_cost.png              # Reviews per day
    ├── efficiency.png              # Efficiency (SRP / Cost)
    ├── comprehensive_metrics.png  # All metrics in one view
    └── hybrid_metrics_detailed.csv # Time-series metrics data
```

## Schedulers Compared

1. **SM-2**: Traditional SuperMemo algorithm (baseline)
2. **Hybrid**: SM-2 + ML adaptive blending
3. **Pure ML**: ML-only predictions (no SM-2)

## Key Findings

- **Interval Optimization**: Hybrid reduces average intervals by 66% vs SM-2
- **Prediction Accuracy**: All schedulers show identical accuracy (MAE = 0.3168)
- **ML Contribution**: Hybrid shows 2% ML confidence (needs more training)

## Metrics Explained

### Recall Curves
Average recall probability at key time points (30, 60, 180, 365 days). Higher is better.

### THR (Target Half-life Reached)
Percentage of items achieving target half-lives (≥180 or ≥365 days). Higher indicates better long-term retention.

### SRP (Summation of Recall Probability)
Cumulative sum of all recall probabilities. Higher indicates better overall learning.

### WTL (Words Total Learned)
Number of items with ≥90% recall for last 3 reviews. Indicates mastery achievement.

### Daily Cost
Average reviews per day. Lower is more efficient (fewer reviews needed).

### Efficiency
SRP ÷ Cost ratio. Higher indicates better recall per review (more efficient learning).

## Regenerating Results

To regenerate all research outputs:

```bash
python scripts/generate_complete_research.py
```

This will:
1. Run scheduler comparison (SM-2, Hybrid, Pure ML)
2. Generate all visualizations
3. Create research report

## Individual Scripts

### Compare Schedulers
```bash
python scripts/compare_schedulers.py research_output/training_data.tsv \
    --hlr-weights research_output/models/HLR_weights.tsv \
    --dhp-params research_output/models/DHP/model.csv \
    --max-samples 1000 \
    --output research_output/comparison_results.csv
```

### Generate Visualizations
```bash
python scripts/visualize_hybrid_metrics.py research_output/comparison_results.csv \
    --output-dir research_output/visualizations
```

## Research Report

See `RESEARCH_REPORT.md` for:
- Detailed methodology
- Complete analysis
- Key findings
- Conclusions and recommendations

## Contact

For questions or issues, refer to the main project documentation.



