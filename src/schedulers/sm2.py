"""
SM-2 Spaced Repetition Scheduler
Integrated with BaseScheduler interface for evaluation framework.
"""

import math
from typing import Dict, List, Optional, Tuple
from .base import (
    BaseScheduler, 
    SchedulerState, 
    ScheduleDecision,
    ReviewRecord,
    exponential_forgetting_curve,
    calculate_half_life_from_recall
)


class SM2Scheduler(BaseScheduler):
    """
    SM-2 algorithm implementation following the official specification.
    
    This implementation is faithful to the original SM-2 algorithm while
    providing additional features for evaluation and research.
    """
    
    # SM-2 constants
    DEFAULT_EF = 2.5
    MIN_EF = 1.3
    FIRST_INTERVAL = 1
    SECOND_INTERVAL = 6
    
    def __init__(self):
        """Initialize SM-2 scheduler."""
        super().__init__()
    
    def schedule_review(self, 
                       user_id: str, 
                       item_id: str, 
                       rating: int,
                       timestamp: float) -> ScheduleDecision:
        """
        Process review and calculate next interval using SM-2.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            rating: Quality score 0-5 (use RatingConverter if needed)
            timestamp: Current timestamp (days)
            
        Returns:
            ScheduleDecision with next review details
        """
        # Get current state
        state = self.get_state(user_id, item_id)
        
        # Extract current parameters
        ease_factor = state.ease_factor
        repetitions = state.repetitions
        interval = state.interval
        
        # Apply SM-2 algorithm
        new_interval, new_repetitions, new_ease_factor = self._sm2_algorithm(
            quality=rating,
            repetitions=repetitions,
            ease_factor=ease_factor,
            interval=interval
        )
        
        # Calculate half-life for recall prediction
        half_life = self._calculate_half_life(new_ease_factor, new_interval)
        
        # Predict recall probability
        p_recall = exponential_forgetting_curve(new_interval, half_life)
        
        # Update state
        state.ease_factor = new_ease_factor
        state.repetitions = new_repetitions
        state.interval = new_interval
        state.last_review = timestamp
        
        # Add to review history
        review_record = ReviewRecord(
            user_id=user_id,
            item_id=item_id,
            rating=rating,
            timestamp=timestamp,
            interval=new_interval,
            ease_factor=new_ease_factor,
            repetitions=new_repetitions
        )
        state.review_history.append(review_record)
        
        # Update scheduler statistics
        self.total_reviews += 1
        if (user_id, item_id) not in self.states:
            self.total_items += 1
        
        # Update state in scheduler
        self.update_state(user_id, item_id, state)
        
        return ScheduleDecision(
            interval=new_interval,
            ease_factor=new_ease_factor,
            repetitions=new_repetitions,
            p_recall=p_recall,
            confidence=1.0  # SM-2 is deterministic
        )
    
    def predict_recall(self, user_id: str, item_id: str, delta_t: float) -> float:
        """
        Predict recall probability after delta_t days.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            delta_t: Days since last review
            
        Returns:
            Predicted recall probability (0-1)
        """
        state = self.get_state(user_id, item_id)
        if state.repetitions == 0:
            return 0.0
        
        half_life = self._calculate_half_life(state.ease_factor, state.interval)
        return exponential_forgetting_curve(delta_t, half_life)
    
    def calculate_half_life(self, user_id: str, item_id: str) -> float:
        """
        Calculate memory half-life for an item.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Half-life in days
        """
        state = self.get_state(user_id, item_id)
        if state.repetitions == 0:
            return 1.0
        
        return self._calculate_half_life(state.ease_factor, state.interval)
    
    def _sm2_algorithm(self, quality: int, repetitions: int, ease_factor: float, interval: int) -> Tuple[int, int, float]:
        """
        Core SM-2 algorithm implementation.
        
        Args:
            quality: Quality rating (0-5)
            repetitions: Current repetition count
            ease_factor: Current ease factor
            interval: Current interval
            
        Returns:
            Tuple of (new_interval, new_repetitions, new_ease_factor)
        """
        # Update ease factor
        if quality >= 3:  # Correct response
            if repetitions == 0:
                new_interval = self.FIRST_INTERVAL
                new_repetitions = 1
            elif repetitions == 1:
                new_interval = self.SECOND_INTERVAL
                new_repetitions = 2
            else:
                new_interval = int(interval * ease_factor)
                new_repetitions = repetitions + 1
            
            # Update ease factor
            new_ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            new_ease_factor = max(new_ease_factor, self.MIN_EF)
        else:  # Incorrect response
            new_interval = self.FIRST_INTERVAL
            new_repetitions = 0
            new_ease_factor = ease_factor  # Don't change EF on failure
        
        return new_interval, new_repetitions, new_ease_factor
    
    def _calculate_half_life(self, ease_factor: float, interval: int) -> float:
        """
        Calculate half-life from ease factor and interval.
        
        Args:
            ease_factor: Current ease factor
            interval: Current interval
            
        Returns:
            Half-life in days
        """
        # Heuristic: half-life ≈ interval / 1.5
        # This is a reasonable approximation for SM-2
        return max(interval / 1.5, 1.0)


class SM2StandaloneCard:
    """
    Standalone SM-2 card implementation for backwards compatibility.
    
    This class provides a simple interface for individual cards
    without the full scheduler infrastructure.
    """
    
    def __init__(self):
        """Initialize a new SM-2 card."""
        self.ease_factor = 2.5
        self.repetitions = 0
        self.interval = 0
    
    def review(self, quality: int) -> int:
        """
        Process a review and return the next interval.
        
        Args:
            quality: Quality rating (0-5)
            
        Returns:
            Days until next review
        """
        # Apply SM-2 algorithm
        if quality >= 3:  # Correct response
            if self.repetitions == 0:
                self.interval = 1
                self.repetitions = 1
            elif self.repetitions == 1:
                self.interval = 6
                self.repetitions = 2
            else:
                self.interval = int(self.interval * self.ease_factor)
                self.repetitions += 1
            
            # Update ease factor
            self.ease_factor += 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
            self.ease_factor = max(self.ease_factor, 1.3)
        else:  # Incorrect response
            self.interval = 1
            self.repetitions = 0
            # Don't change ease factor on failure
        
        return self.interval
    
    def __str__(self) -> str:
        """String representation of the card state."""
        return f"SM2Card(EF={self.ease_factor:.2f}, reps={self.repetitions}, interval={self.interval})"