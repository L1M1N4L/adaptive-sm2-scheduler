"""
Scheduler Package for Adaptive Spaced Repetition Algorithms

This package implements various spaced repetition scheduling algorithms
for Braille literacy optimization through IoT tactile flashcards.

Algorithms:
- SM-2: Traditional SuperMemo algorithm (baseline)
- FSRS: Free Spaced Repetition Scheduler with stability modeling
- ML-Based: Machine learning interval prediction
- Threshold: Fixed half-life threshold scheduling
- Hybrid: SM-2 + AI combination with dynamic blending
"""

from .base import (
    BaseScheduler,
    SchedulerState,
    ScheduleDecision,
    ReviewRecord,
    RatingConverter,
    exponential_forgetting_curve,
    calculate_half_life_from_recall
)

from .sm2 import SM2Scheduler, SM2StandaloneCard
from .hybrid import HybridScheduler

__all__ = [
    'BaseScheduler',
    'SchedulerState', 
    'ScheduleDecision',
    'ReviewRecord',
    'RatingConverter',
    'SM2Scheduler',
    'SM2StandaloneCard',
    'HybridScheduler',
    'exponential_forgetting_curve',
    'calculate_half_life_from_recall'
]

__version__ = "1.0.0"
