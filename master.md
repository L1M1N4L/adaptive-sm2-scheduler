# Master Prompt: Cognitive-AI Approaches to Adaptive Scheduling in Braille Memory Retention Using Hybrid SM-2 Models

## Project Context

You are developing a research proposal that addresses the critical challenge of Braille literacy through advanced spaced repetition scheduling algorithms. Only 10-25% of legally blind individuals in the U.S. are fluent in Braille, yet those who achieve fluency demonstrate significantly better employment and literacy outcomes. This research aims to bridge the gap between traditional algorithmic approaches and modern machine learning techniques to optimize Braille learning through IoT-enabled tactile flashcards.

## Research Problem Statement

Current spaced repetition scheduling algorithms face fundamental trade-offs that limit educational effectiveness:

**SM-2 Algorithm Limitations:**
- Transparent and interpretable
- Static, non-adaptive to individual learning patterns
- Limited personalization capabilities
- Uniform treatment of all items and learners

**Machine Learning Approaches:**
- Adaptive and personalized
- Opaque decision-making processes
- Risk of overfitting
- Lack of transparency and interpretability
- Computationally complex

**FSRS Constraints:**
- Provides predictive capabilities
- Fixed parametric forms limit flexibility
- Cannot fully adapt to diverse learning patterns

## Core Research Objectives

1. **Develop baseline (SM-2), ML, and hybrid scheduling methods** for Braille character retention
2. **Compare scheduling policies** using SSP-MMC-Plus evaluation metrics
3. **Apply optimized scheduling** to Braille literacy through IoT tactile flashcard systems

## Background Knowledge

### Pattern Recognition in Braille Learning
Braille literacy operates as a pattern recognition system where tactile dot combinations map to letters and symbols, functioning similarly to visual flashcard learning through pattern-based memory formation.

### Spaced Repetition Fundamentals
Spaced repetition reliably improves long-term retention across domains, reducing forgetting curves and enhancing learning efficiency through strategic review timing.

### Current Algorithmic Landscape
- Traditional algorithms (SM-2) offer transparency but rigid limitations
- Pure ML approaches provide adaptability but suffer from black-box operations
- Hybrid solutions remain unexplored in tactile learning domains

## Proposed Solution: Hybrid SM-2 + AI Scheduler

### Algorithm Components

**1. SM-2 Baseline Calculation**
- Each review outcome (quality score q) updates ease factor (EF) and repetition count
- Intervals grow approximately exponentially with EF
- Provides interpretable, rule-based scheduling foundation
- Calculate next review interval: I_sm2

**2. ML-Based Half-Life Prediction**
- Estimate learner-specific memory half-life
- Derive predicted interval: I_ML
- Capture individual learning dynamics and patterns
- Account for item difficulty and learner proficiency

**3. Hybrid Blending Strategy**
```
I_hybrid = β × I_sm2 + (1 - β) × I_ML
```
- β = blending factor (0 to 1)
- Dynamically adjust β based on:
  - Model confidence levels
  - Learner's stage in curriculum
  - Historical prediction accuracy
  - Data availability per user

**4. Adaptive Optimization**
- Continuously refine β parameter
- Balance interpretability with personalization
- Maintain transparency while improving outcomes

## Methodology

### Data Sources

**Primary Dataset: FSRS-Anki-20k (HuggingFace)**
- Review histories from 20,000 Anki users
- Millions of review entries
- Rich behavioral patterns

**Secondary Dataset: Harvard Dataverse Spaced Repetition Dataset**
- Additional learner logs
- Cross-validation support
- Diverse learning contexts

**Data Schema Mapping (SSP-MMC-Plus):**
- u: user ID
- w: word/item ID
- i: item index
- d: difficulty level
- t_history: timestamp history
- r_history: recall history (binary outcomes)
- delta_t: time since last review
- r: current recall outcome
- p_recall: predicted recall probability
- total_cnt: total review count

### Memory Half-Life Concept

**Definition:** Time duration until recall probability decreases to 0.5 (50%)

**Importance:**
- Captures memory durability beyond single recall accuracy
- Directly informs optimal review scheduling (review before half-life expires)
- Provides fair, comparable metric across different algorithms
- Models forgetting curves at individual learner level

### Evaluation Pipeline

**Phase 1: Fit Memory Model**
- Train half-life regression models on historical data
- Develop ML predictor for recall probability estimation
- Validate model accuracy on held-out data

**Phase 2: Policy Replay Simulation**
Apply five scheduling policies in simulator:
1. **SM-2**: Traditional spaced repetition baseline
2. **FSRS**: Free Spaced Repetition Scheduler with stability modeling
3. **ML-only**: Pure machine learning interval prediction
4. **Threshold**: Fixed half-life threshold scheduling
5. **Hybrid**: Proposed SM-2 + AI combination

Each policy selects review intervals based on its logic. Simulator generates probabilistic recall outcomes.

**Phase 3: Forward Simulation**
- Extend review schedules up to 365 days
- Generate synthetic recall data for long-term outcome prediction
- Model realistic learning trajectories

**Phase 4: Metric Computation**
Calculate performance at checkpoints (30, 60, 180, 365 days)

**Phase 5: Cross-Dataset Validation**
- Train on FSRS-Anki-20k → Test on Harvard Dataverse
- Train on Harvard → Test on FSRS-Anki-20k
- Assess generalization and robustness

### Evaluation Metrics

**1. Recall Curves**
- Average recall probability at 30, 60, 180, and 365 days
- Tracks long-term retention effectiveness

**2. THR (Target Half-life Reached)**
- Percentage of items achieving half-life ≥ 180 days
- Percentage of items achieving half-life ≥ 365 days
- Measures durable learning outcomes

**3. SRP (Summation of Recall Probability)**
```
SRP = Σ(i=1 to N) p_i
```
- Aggregate retention across all items
- Overall learning effectiveness measure

**4. WTL (Words Total Learned)**
- Count of items with recall ≥ 90% for last three consecutive reviews
- Mastery achievement metric

**5. Daily Cost**
- Average number of reviews per day
- Study burden quantification

**6. Efficiency**
```
Efficiency = SRP ÷ Daily Cost
```
- Learning effectiveness per unit of study effort
- Optimization target for practical deployment

## Expected Contributions

1. **Comparative Algorithmic Study**
   - Systematic evaluation of five scheduling approaches
   - Empirical performance benchmarks on real learning data
   - Identification of trade-offs and optimal use cases

2. **Novel Hybrid Scheduler**
   - Development of interpretable yet adaptive scheduling system
   - Blending strategy with dynamic confidence-based weighting
   - Balance between transparency and personalization

3. **Half-Life Evaluation Framework**
   - Demonstration of half-life as key evaluation metric
   - Standardized approach for comparing scheduling algorithms
   - Direct connection to cognitive science principles

4. **Braille Literacy Application Roadmap**
   - Practical implementation pathway via IoT tactile flashcards
   - Hardware integration specifications
   - User experience considerations for visually impaired learners

5. **Open-Source Implementation**
   - Replicable methodology and code
   - Community contribution to accessibility technology
   - Foundation for future adaptive learning research

## Implementation Roadmap

### Phase 1: Baseline Implementation (Weeks 1-3)
- Implement SM-2 algorithm from scratch
- Implement FSRS with stability modeling
- Validate against published benchmarks
- Set up evaluation framework

### Phase 2: ML Model Training (Weeks 4-7)
- Feature engineering from review histories
- Train half-life regression models (XGBoost, Neural Networks, Random Forest)
- Hyperparameter optimization
- Cross-validation and model selection

### Phase 3: Hybrid Scheduler Development (Weeks 8-10)
- Implement blending mechanism
- Develop β adjustment strategies
- Test confidence-based and stage-based weighting
- Optimize for interpretability

### Phase 4: Comprehensive Evaluation (Weeks 11-14)
- Run SSP-MMC-Plus metric evaluation pipeline
- Generate recall curves and THR statistics
- Calculate SRP, WTL, Daily Cost, and Efficiency
- Cross-dataset validation
- Statistical significance testing

### Phase 5: IoT Integration Planning (Weeks 15-16)
- Design tactile flashcard hardware specifications
- Develop API for scheduling algorithm integration
- Plan user study protocol for Braille learners
- Document deployment considerations

## Challenges and Mitigation Strategies

### Challenge 1: ML Model Validity
**Issue:** ML predictor may not simulate human brain 1:1 (noted by Ken)
**Mitigation:** 
- Use ML as augmentation, not replacement of SM-2
- Maintain interpretable baseline through hybrid approach
- Validate on real human learning data, not synthetic
- Conservative blending factor (start with β > 0.5 favoring SM-2)

### Challenge 2: Overfitting Risk
**Mitigation:**
- Regularization techniques (L1/L2, dropout)
- Cross-validation across datasets
- Monitor train vs. test performance gaps
- Use ensemble methods to reduce variance

### Challenge 3: Cold Start Problem
**Mitigation:**
- Default to SM-2 for new users (β = 1.0)
- Gradually shift to hybrid as data accumulates
- Transfer learning from similar users
- Population-level priors for initialization

### Challenge 4: Computational Complexity
**Mitigation:**
- Offline batch training of ML models
- Lightweight inference for real-time scheduling
- Caching of predictions
- Edge computing on IoT devices

## Success Criteria

**Technical Success:**
- Hybrid scheduler achieves ≥10% improvement in SRP vs. SM-2
- Maintains ≥80% interpretability (measurable through user studies)
- Reduces Daily Cost while maintaining recall performance
- Generalizes across both datasets (cross-validation AUC ≥ 0.75)

**Application Success:**
- Roadmap demonstrates feasibility for IoT tactile implementation
- Algorithm runs efficiently on resource-constrained devices
- User study protocol approved for visually impaired participants
- Open-source code repository with documentation

## Ethical Considerations

- **Accessibility First:** Design prioritizes needs of visually impaired learners
- **Privacy Protection:** Anonymize all learning data, no PII collection
- **Transparency:** Provide explanations for scheduling decisions
- **Autonomy:** Allow users to override algorithm recommendations
- **Inclusivity:** Test across diverse age groups and learning backgrounds

## Deliverables

1. **Research Paper:** Full comparative study with results and analysis
2. **Open-Source Codebase:** Documented implementation of all algorithms
3. **Dataset:** Processed SSP-MMC-Plus formatted data (if permissible)
4. **Technical Report:** IoT tactile flashcard integration specifications
5. **Presentation:** Conference-ready slides and demonstration

## Key References

**Jankowski, J. (2022).** Application of a computer to improve the results obtained in working with the SuperMemo method. SuperMemo. [https://www.supermemo.com/en/blog/application-of-a-computer-to-improve-the-results-obtained-in-working-with-the-supermemo-method](https://www.supermemo.com/en/blog/application-of-a-computer-to-improve-the-results-obtained-in-working-with-the-supermemo-method)

**Su, J., Ye, J., Nie, L., Cao, Y., & Chen, Y. (2023).** Optimizing spaced repetition schedule by capturing the dynamics of memory. IEEE Transactions on Knowledge and Data Engineering, 35(10), 10085–10097. [https://doi.org/10.1109/tkde.2023.3251721](https://doi.org/10.1109/tkde.2023.3251721)

**Ye, J., Su, J., & Cao, Y. (2022).** A stochastic shortest path algorithm for optimizing spaced repetition scheduling. Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4381–4390. [https://doi.org/10.1145/3534678.3539081](https://doi.org/10.1145/3534678.3539081)

---

## Usage Instructions for This Master Prompt

This master prompt serves as a comprehensive specification for the research project. Use it to:

1. **Guide Development:** Reference specific sections when implementing components
2. **Maintain Consistency:** Ensure all work aligns with stated objectives and methodology
3. **Communicate with Stakeholders:** Share relevant sections with advisors, collaborators
4. **Generate Documentation:** Extract content for papers, reports, and presentations
5. **Track Progress:** Use roadmap phases as checkpoints for project management

When working with AI assistants or team members, provide this prompt as context to ensure everyone understands the full scope, technical approach, and success criteria of the project.