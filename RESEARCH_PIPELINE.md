# Research Pipeline: Train, Compare, and Visualize

This document describes the complete research pipeline for training models, comparing schedulers, and creating visualizations.

## Quick Start

Run the complete pipeline:

```bash
python scripts/run_research_pipeline.py fsrs_anki_20k_part_1.csv \
    --max-cards 5000 \
    --max-samples 1000 \
    --output ./research_output
```

## Pipeline Steps

### Step 1: Prepare Training Data
Converts FSRS-Anki CSV to SSP-MMC-Plus TSV format:
- Extracts review histories
- Calculates half-lives and recall probabilities
- Formats for model training

### Step 2: Train Models
Trains HLR and DHP models:
- **HLR (Half-Life Regression)**: Learns weights for feature-based prediction
- **DHP (Difficulty-Half-life-Prediction)**: Learns parameters for difficulty-based prediction

### Step 3: Compare Schedulers
Evaluates multiple schedulers:
- **SM-2**: Baseline traditional algorithm
- **Hybrid (trained)**: SM-2 + ML with trained models
- **Hybrid (untrained)**: SM-2 + ML with default weights

Metrics calculated:
- Mean Absolute Error (MAE) for recall prediction
- Root Mean Squared Error (RMSE)
- Average intervals and half-lives
- ML confidence scores

### Step 4: Create Visualizations
Generates research-quality plots:
- **prediction_accuracy.png**: MAE, scatter plots, error distributions
- **ml_contribution.png**: ML confidence analysis
- **metrics_comparison.png**: Side-by-side metric comparisons
- **learning_curves.png**: Performance over review number

### Step 5: Generate Summary
Creates summary report with all results.

## Output Files

After running the pipeline, you'll have:

```
research_output/
├── training_data.tsv          # Converted training data
├── models/
│   ├── HLR_weights.tsv        # Trained HLR model weights
│   └── DHP/
│       └── model.csv           # Trained DHP parameters
├── comparison_results.csv      # Detailed comparison results
├── comparison_metrics.csv      # Summary metrics
└── visualizations/
    ├── prediction_accuracy.png
    ├── ml_contribution.png
    ├── metrics_comparison.png
    └── learning_curves.png
```

## Individual Scripts

You can also run scripts individually:

### Prepare Data
```bash
python scripts/prepare_training_data.py \
    fsrs_anki_20k_part_1.csv \
    training_data.tsv \
    --max-cards 5000
```

### Train Models
```bash
# Train HLR
python scripts/train_hybrid_models.py \
    training_data.tsv \
    --output-dir ./models \
    --hlr-only

# Train DHP
python scripts/train_hybrid_models.py \
    training_data.tsv \
    --output-dir ./models \
    --dhp-only
```

### Compare Schedulers
```bash
python scripts/compare_schedulers.py \
    training_data.tsv \
    --hlr-weights ./models/HLR_weights.tsv \
    --dhp-params ./models/DHP/model.csv \
    --max-samples 1000 \
    --output ./comparison_results.csv
```

### Visualize
```bash
python scripts/visualize_comparison.py \
    comparison_results.csv \
    comparison_metrics.csv \
    --output-dir ./visualizations
```

## Research Metrics

The pipeline calculates:

1. **Prediction Accuracy**
   - MAE: Mean Absolute Error
   - RMSE: Root Mean Squared Error
   - Error distributions

2. **Scheduling Performance**
   - Average review intervals
   - Memory half-lives
   - Predicted recall probabilities

3. **ML Contribution**
   - ML confidence scores
   - Contribution over review number
   - Interval adjustments

4. **Learning Curves**
   - Performance over time
   - Adaptation patterns
   - Convergence analysis

## Visualization Details

### Prediction Accuracy
- Bar chart of MAE by scheduler
- Scatter plot: predicted vs actual recall
- Error distribution histograms
- Interval comparison

### ML Contribution
- ML confidence distribution
- Confidence over review number
- Interval difference (Hybrid vs SM-2)
- Confidence vs prediction quality

### Metrics Comparison
- Side-by-side bar charts
- Statistical comparisons
- Performance rankings

### Learning Curves
- Interval growth over reviews
- Half-life development
- Prediction accuracy trends
- Recall probability evolution

## Interpreting Results

### Good Performance Indicators
- **Low MAE**: Accurate recall predictions
- **High ML confidence**: ML models contributing meaningfully
- **Adaptive intervals**: Intervals adjust based on performance
- **Stable learning curves**: Consistent improvement over time

### Comparison Insights
- **Hybrid vs SM-2**: Shows benefit of ML integration
- **Trained vs Untrained**: Demonstrates value of model training
- **ML confidence trends**: Indicates when ML becomes most useful

## Troubleshooting

### Training Fails
- Check data format matches expected TSV structure
- Ensure sufficient data (at least 1000 samples)
- Verify all required columns present

### Low ML Confidence
- Train on more data
- Check model weights loaded correctly
- Verify models completed training

### Visualization Errors
- Install required packages: `pip install matplotlib seaborn`
- Check data files exist
- Verify CSV format is correct

## Next Steps

1. **Analyze Results**: Review metrics and visualizations
2. **Tune Parameters**: Adjust beta, confidence thresholds
3. **Scale Up**: Train on full dataset
4. **Publish**: Use visualizations in research paper

## Citation

If using this pipeline for research, please cite:
- SSP-MMC-Plus framework
- FSRS-Anki-20k dataset
- Your hybrid scheduler implementation

