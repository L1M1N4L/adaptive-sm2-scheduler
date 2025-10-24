# Adaptive SM-2 Scheduler

A professional implementation of the SM-2 spaced repetition algorithm for Braille learning research.

## What is this?

This project implements the SM-2 algorithm (used by Anki and SuperMemo) for scheduling when to review Braille characters. It helps optimize learning by showing flashcards at the right time to maximize memory retention.

## Getting Started

### 1. Install Python
Make sure you have Python 3.7 or newer installed on your computer.

### 2. Download the Project
```bash
git clone <repository-url>
cd adaptive-sm2-scheduler
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Test Everything Works
```bash
python tests/test_sm2_comprehensive.py
```

## How to Use

### Basic Usage
```python
from src.schedulers import SM2Scheduler

# Create a scheduler
scheduler = SM2Scheduler()

# Review a Braille character
decision = scheduler.schedule_review("learner_001", "braille_a", 4, 0.0)
print(f"Review again in {decision.interval} days")
```

### Run Evaluation
```bash
python scripts/comprehensive_evaluation.py
```

### See Examples
```bash
python examples/sm2_demo.py
```

## Project Structure

```
adaptive-sm2-scheduler/
├── src/schedulers/          # The main SM-2 algorithm code
├── scripts/                 # Tools for running evaluations
├── evaluation/              # Results and evaluation framework
├── tests/                   # Tests to make sure everything works
├── examples/                # Example code showing how to use it
├── docs/                    # Detailed documentation
└── data/                    # Where datasets will be stored
```

## What's Inside

- **SM-2 Algorithm**: The core spaced repetition algorithm
- **Evaluation Tools**: Tools to test how well the algorithm works
- **Documentation**: Complete guides and API reference
- **Examples**: Ready-to-run code examples

## Current Status

✅ **Phase 1 Complete**: SM-2 algorithm working perfectly  
🔄 **Phase 4 In Progress**: Evaluation framework running  
⏳ **Next**: Ready for real Braille learning data

## Documentation

For detailed information, see the [docs/](docs/) folder:
- [API Reference](docs/api_reference.md) - Complete code documentation
- [Algorithm Details](docs/algorithm_specification.md) - How SM-2 works
- [Research Plan](docs/research_roadmap.md) - Future development phases

## Requirements

- Python 3.7 or newer
- NumPy (installed automatically)

## Research Purpose

This is part of research to improve Braille literacy by optimizing when people review Braille characters. The goal is to help people learn Braille faster and remember it longer.

## License

Open source - see project documentation for details."# adaptive-sm2-scheduler" 
