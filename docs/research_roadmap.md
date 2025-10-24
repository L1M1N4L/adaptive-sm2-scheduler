# Research Roadmap

## Project Overview

This project implements "Cognitive-AI Approaches to Adaptive Scheduling in Braille Memory Retention Using Hybrid SM-2 Models" - a research initiative aimed at improving Braille literacy rates through optimized spaced repetition algorithms.

## Research Problem

Current spaced repetition scheduling algorithms face fundamental trade-offs that limit educational effectiveness:

- **SM-2 Algorithm**: Transparent and interpretable but static and non-adaptive
- **Machine Learning Approaches**: Adaptive and personalized but opaque and computationally complex
- **FSRS**: Provides predictive capabilities but has fixed parametric forms

## Research Objectives

1. Develop baseline (SM-2), ML, and hybrid scheduling methods for Braille character retention
2. Compare scheduling policies using SSP-MMC-Plus evaluation metrics
3. Apply optimized scheduling to Braille literacy through IoT tactile flashcard systems

## Implementation Phases

### Phase 1: Baseline Implementation (Weeks 1-3)

**Status**: COMPLETE

**Deliverables**:
- SM-2 algorithm implementation from scratch
- Professional project structure and organization
- Comprehensive test suite with 100% pass rate
- SSP-MMC-Plus evaluation framework integration
- Complete documentation suite

**Current Status**: SM-2 implementation complete, tested, and professionally organized

### Phase 2: ML Model Training (Weeks 4-7)

**Status**: PENDING

**Deliverables**:
- Feature engineering from review histories
- Train half-life regression models (XGBoost, Neural Networks, Random Forest)
- Hyperparameter optimization
- Cross-validation and model selection

**Dependencies**: FSRS-Anki-20k dataset download

### Phase 3: Hybrid Scheduler Development (Weeks 8-10)

**Status**: PENDING

**Deliverables**:
- Implement blending mechanism
- Develop β adjustment strategies
- Test confidence-based and stage-based weighting
- Optimize for interpretability

**Key Innovation**: Dynamic blending of SM-2 and ML predictions

### Phase 4: Comprehensive Evaluation (Weeks 11-14)

**Status**: IN PROGRESS

**Deliverables**:
- SSP-MMC-Plus evaluation framework integrated
- Automated evaluation pipeline working
- Sample data evaluation completed
- Results generation and reporting system
- Ready for real dataset evaluation

**Current Status**: Evaluation framework working with sample data, ready for real datasets

### Phase 5: IoT Integration Planning (Weeks 15-16)

**Status**: PENDING

**Deliverables**:
- Design tactile flashcard hardware specifications
- Develop API for scheduling algorithm integration
- Plan user study protocol for Braille learners
- Document deployment considerations

## Evaluation Methodology

### SSP-MMC-Plus Metrics

1. **Recall Curves**: Average recall probability at 30, 60, 180, and 365 days
2. **THR (Target Half-life Reached)**: Percentage of items achieving half-life ≥ 180/365 days
3. **SRP (Summation of Recall Probability)**: Aggregate retention across all items
4. **WTL (Words Total Learned)**: Count of items with recall ≥ 90% for last three consecutive reviews
5. **Daily Cost**: Average number of reviews per day
6. **Efficiency**: SRP ÷ Daily Cost

### Data Sources

- **Primary**: FSRS-Anki-20k (HuggingFace) - 20,000 Anki users, millions of review entries
- **Secondary**: Harvard Dataverse Spaced Repetition Dataset - additional learner logs

### Evaluation Pipeline

1. **Fit Memory Model**: Train half-life regression models on historical data
2. **Policy Replay Simulation**: Apply five scheduling policies in simulator
3. **Forward Simulation**: Extend review schedules up to 365 days
4. **Metric Computation**: Calculate performance at checkpoints
5. **Cross-Dataset Validation**: Train/test on different datasets

## Expected Contributions

1. **Comparative Algorithmic Study**: Systematic evaluation of five scheduling approaches
2. **Novel Hybrid Scheduler**: Interpretable yet adaptive scheduling system
3. **Half-Life Evaluation Framework**: Standardized approach for comparing algorithms
4. **Braille Literacy Application Roadmap**: Practical implementation pathway
5. **Open-Source Implementation**: Replicable methodology and code

## Success Criteria

### Technical Success

- Hybrid scheduler achieves ≥10% improvement in SRP vs. SM-2
- Maintains ≥80% interpretability (measurable through user studies)
- Reduces Daily Cost while maintaining recall performance
- Generalizes across both datasets (cross-validation AUC ≥ 0.75)

### Application Success

- Roadmap demonstrates feasibility for IoT tactile implementation
- Algorithm runs efficiently on resource-constrained devices
- User study protocol approved for visually impaired participants
- Open-source code repository with documentation

## Challenges and Mitigation Strategies

### Challenge 1: ML Model Validity

**Issue**: ML predictor may not simulate human brain 1:1

**Mitigation**:
- Use ML as augmentation, not replacement of SM-2
- Maintain interpretable baseline through hybrid approach
- Validate on real human learning data, not synthetic
- Conservative blending factor (start with β > 0.5 favoring SM-2)

### Challenge 2: Overfitting Risk

**Mitigation**:
- Regularization techniques (L1/L2, dropout)
- Cross-validation across datasets
- Monitor train vs. test performance gaps
- Use ensemble methods to reduce variance

### Challenge 3: Cold Start Problem

**Mitigation**:
- Default to SM-2 for new users (β = 1.0)
- Gradually shift to hybrid as data accumulates
- Transfer learning from similar users
- Population-level priors for initialization

## Ethical Considerations

- **Accessibility First**: Design prioritizes needs of visually impaired learners
- **Privacy Protection**: Anonymize all learning data, no PII collection
- **Transparency**: Provide explanations for scheduling decisions
- **Autonomy**: Allow users to override algorithm recommendations
- **Inclusivity**: Test across diverse age groups and learning backgrounds

## Deliverables

1. **Research Paper**: Full comparative study with results and analysis
2. **Open-Source Codebase**: Documented implementation of all algorithms
3. **Dataset**: Processed SSP-MMC-Plus formatted data (if permissible)
4. **Technical Report**: IoT tactile flashcard integration specifications
5. **Presentation**: Conference-ready slides and demonstration

## Current Status

- **Phase 1**: SM-2 implementation complete
- **Next Steps**: Download FSRS-Anki-20k dataset and begin evaluation framework
- **Architecture**: Scalable foundation ready for all phases
- **Code Quality**: Professional implementation with comprehensive testing
