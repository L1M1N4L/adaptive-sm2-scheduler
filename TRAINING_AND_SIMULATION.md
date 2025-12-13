# Training and Simulation Guide

This guide explains how to train ML models and simulate learning with the hybrid scheduler using the SSP-MMC-Plus framework.

## Overview

The hybrid scheduler combines SM-2 with ML models (HLR and DHP). To get the best performance, you need to:

1. **Convert** FSRS-Anki dataset to SSP-MMC-Plus format
2. **Train** HLR and DHP models on your data
3. **Simulate** learning with the trained hybrid scheduler

## Quick Start

### Full Pipeline (Recommended)

Run the complete pipeline in one command:

```bash
python scripts/run_full_pipeline.py fsrs_anki_20k_part_1.csv --max-cards 5000
```

This will:
- Convert the data
- Train both HLR and DHP models
- Run simulation with the hybrid scheduler

### Step-by-Step

#### Step 1: Convert Data

Convert FSRS-Anki CSV to SSP-MMC-Plus TSV format:

```bash
python scripts/prepare_training_data.py \
    fsrs_anki_20k_part_1.csv \
    evaluation/SSP-MMC-Plus/tmp/training_data.tsv \
    --max-cards 5000
```

#### Step 2: Train Models

Train HLR and DHP models:

```bash
python scripts/train_hybrid_models.py \
    evaluation/SSP-MMC-Plus/tmp/training_data.tsv \
    --output-dir ./evaluation/SSP-MMC-Plus/tmp
```

Or train individually:

```bash
# Train only HLR
python scripts/train_hybrid_models.py \
    evaluation/SSP-MMC-Plus/tmp/training_data.tsv \
    --hlr-only

# Train only DHP
python scripts/train_hybrid_models.py \
    evaluation/SSP-MMC-Plus/tmp/training_data.tsv \
    --dhp-only
```

#### Step 3: Simulate

Run simulation with trained models:

```bash
python scripts/simulate_hybrid.py \
    evaluation/SSP-MMC-Plus/tmp/training_data.tsv \
    --hlr-weights ./evaluation/SSP-MMC-Plus/tmp/HLR_weights.tsv \
    --dhp-params ./evaluation/SSP-MMC-Plus/tmp/DHP/model.csv \
    --max-cards 1000 \
    --output ./simulation/hybrid_results.tsv
```

## Using Trained Models in Code

After training, use the trained models in your hybrid scheduler:

```python
from src.schedulers.hybrid import HybridScheduler

# Initialize with trained models
hybrid = HybridScheduler(
    use_hlr=True,
    use_dhp=True,
    use_rnn=False,
    hlr_weights_path='./evaluation/SSP-MMC-Plus/tmp/HLR_weights.tsv',
    dhp_params_path='./evaluation/SSP-MMC-Plus/tmp/DHP/model.csv'
)

# Use the scheduler
decision = hybrid.schedule_review(
    user_id="user1",
    item_id="item1",
    rating=4,  # SM-2 quality (0-5)
    timestamp=10.0  # Days
)

print(f"Interval: {decision.interval} days")
print(f"ML Confidence: {decision.confidence:.2%}")
print(f"Predicted Recall: {decision.p_recall:.2%}")
```

## Expected Output Files

After running the pipeline, you'll have:

- `evaluation/SSP-MMC-Plus/tmp/training_data.tsv` - Converted training data
- `evaluation/SSP-MMC-Plus/tmp/HLR_weights.tsv` - Trained HLR model weights
- `evaluation/SSP-MMC-Plus/tmp/DHP/model.csv` - Trained DHP parameters
- `simulation/hybrid_results.tsv` - Simulation results
- `simulation/hybrid_metrics.tsv` - Daily metrics

## Performance Improvement

With trained models, you should see:

- **Higher ML confidence**: 20-60% (vs 0-1% with untrained models)
- **Better predictions**: ML models contribute meaningfully to interval decisions
- **Adaptive blending**: Beta increases with review history, giving more weight to ML

## Troubleshooting

### Models not loading
- Check that file paths are correct
- Ensure models were trained successfully
- Check file permissions

### Low ML confidence
- Train on more data (increase `--max-cards`)
- Ensure models completed training
- Check that weights/parameters files exist

### Simulation errors
- Verify training data format
- Check that all required columns are present
- Ensure sufficient memory for large datasets

## Next Steps

1. **Evaluate performance**: Compare hybrid vs SM-2 vs pure ML
2. **Tune parameters**: Adjust beta growth rate, confidence thresholds
3. **Scale up**: Train on full dataset for production use
4. **Integrate**: Use trained models in your application



