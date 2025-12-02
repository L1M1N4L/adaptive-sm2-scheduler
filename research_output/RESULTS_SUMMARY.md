# Research Results Summary

## Location of Results

All results and visualizations are located in: **`research_output/`**

### Files Generated

1. **Training Data**
   - `training_data.tsv` - Converted FSRS-Anki data in SSP-MMC-Plus format

2. **Comparison Results**
   - `comparison_results.csv` - Detailed results for each scheduler evaluation
   - `comparison_metrics.csv` - Summary metrics comparing all schedulers

3. **Visualizations**
   - `visualizations/prediction_accuracy.png` - Prediction accuracy analysis
   - `visualizations/ml_contribution.png` - ML model contribution analysis (if hybrid data available)
   - `visualizations/metrics_comparison.png` - Side-by-side metric comparisons
   - `visualizations/learning_curves.png` - Performance over review number

## Quick Access

### View Results
```bash
# View metrics summary
cat research_output/comparison_metrics.csv

# View detailed results (first 20 rows)
head -20 research_output/comparison_results.csv
```

### View Visualizations
Open the PNG files in `research_output/visualizations/`:
- Windows: `start research_output/visualizations/prediction_accuracy.png`
- Mac/Linux: `open research_output/visualizations/prediction_accuracy.png`

## Results Overview

Based on the comparison metrics:

### Schedulers Compared
1. **SM-2**: Baseline traditional algorithm
2. **Hybrid**: SM-2 + ML (with trained models if available)
3. **Hybrid-untrained**: SM-2 + ML (with default weights)

### Key Metrics
- **MAE (Mean Absolute Error)**: Lower is better for prediction accuracy
- **Average Interval**: Days between reviews
- **Average Half-life**: Memory retention duration
- **ML Confidence**: Percentage of ML contribution (0-100%)

## Next Steps

1. **Review Visualizations**: Check the PNG files in `visualizations/` folder
2. **Analyze Metrics**: Review `comparison_metrics.csv` for quantitative comparisons
3. **Examine Details**: Look at `comparison_results.csv` for per-review analysis
4. **Train Models**: To improve ML confidence, train models on more data:
   ```bash
   python scripts/train_hybrid_models.py research_output/training_data.tsv --output-dir research_output/models
   ```

## File Structure

```
research_output/
├── training_data.tsv              # Input training data
├── comparison_results.csv         # Detailed comparison (all reviews)
├── comparison_metrics.csv         # Summary metrics
├── models/                        # Trained model files (if training completed)
│   ├── HLR_weights.tsv
│   └── DHP/model.csv
└── visualizations/                # Research visualizations
    ├── prediction_accuracy.png
    ├── ml_contribution.png
    ├── metrics_comparison.png
    └── learning_curves.png
```

