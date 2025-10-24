"""
Base Scheduler Interface

This module defines the abstract base class for all spaced repetition schedulers
and provides common data structures and utilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import math


@dataclass
class ReviewRecord:
    """Record of a single review session."""
    user_id: str
    item_id: str
    rating: int  # 0-5 scale (SM-2 quality)
    timestamp: float  # Days since epoch
    interval: int  # Days until next review
    ease_factor: float  # Current ease factor
    repetitions: int  # Number of successful repetitions


@dataclass
class SchedulerState:
    """State of a scheduler for a specific user-item pair."""
    ease_factor: float = 2.5
    repetitions: int = 0
    interval: int = 0
    last_review: float = 0.0
    review_history: List[ReviewRecord] = None
    
    def __post_init__(self):
        if self.review_history is None:
            self.review_history = []


@dataclass
class ScheduleDecision:
    """Decision made by a scheduler for a review."""
    interval: int  # Days until next review
    ease_factor: float  # Updated ease factor
    repetitions: int  # Updated repetition count
    p_recall: float  # Predicted recall probability
    confidence: float = 0.0  # Confidence in the decision


class RatingConverter:
    """Utility class for converting between different rating systems."""
    
    @staticmethod
    def anki_to_sm2(anki_rating: int) -> int:
        """
        Convert Anki rating (1-4) to SM-2 quality (0-5).
        
        Args:
            anki_rating: Anki rating (1=Again, 2=Hard, 3=Good, 4=Easy)
            
        Returns:
            SM-2 quality (0-5)
        """
        conversion_map = {1: 0, 2: 3, 3: 4, 4: 5}
        return conversion_map.get(anki_rating, 3)
    
    @staticmethod
    def sm2_to_anki(sm2_quality: int) -> int:
        """
        Convert SM-2 quality (0-5) to Anki rating (1-4).
        
        Args:
            sm2_quality: SM-2 quality (0-5)
            
        Returns:
            Anki rating (1-4)
        """
        if sm2_quality <= 2:
            return 1  # Again
        elif sm2_quality == 3:
            return 2  # Hard
        elif sm2_quality == 4:
            return 3  # Good
        else:  # sm2_quality == 5
            return 4  # Easy


def exponential_forgetting_curve(interval: float, half_life: float) -> float:
    """
    Calculate recall probability using exponential forgetting curve.
    
    Args:
        interval: Time interval in days
        half_life: Memory half-life in days
        
    Returns:
        Recall probability (0-1)
    """
    if half_life <= 0:
        return 0.0
    return 2 ** (-interval / half_life)


def calculate_half_life_from_recall(recall_prob: float, interval: float) -> float:
    """
    Calculate half-life from recall probability and interval.
    
    Args:
        recall_prob: Recall probability (0-1)
        interval: Time interval in days
        
    Returns:
        Half-life in days
    """
    if recall_prob <= 0 or recall_prob >= 1 or interval <= 0:
        return 1.0
    return -interval / math.log2(recall_prob)


class BaseScheduler(ABC):
    """
    Abstract base class for all spaced repetition schedulers.
    
    This class defines the common interface that all schedulers must implement,
    ensuring consistency and enabling easy comparison between different algorithms.
    """
    
    def __init__(self):
        """Initialize the scheduler."""
        self.states: Dict[Tuple[str, str], SchedulerState] = {}
        self.total_reviews = 0
        self.total_items = 0
    
    @abstractmethod
    def schedule_review(self, user_id: str, item_id: str, rating: int, timestamp: float) -> ScheduleDecision:
        """
        Schedule the next review for an item.
        
        Args:
            user_id: Unique identifier for the user
            item_id: Unique identifier for the item
            rating: Review rating (0-5 scale)
            timestamp: Review timestamp in days since epoch
            
        Returns:
            ScheduleDecision with next review details
        """
        pass
    
    @abstractmethod
    def predict_recall(self, user_id: str, item_id: str, delta_t: float) -> float:
        """
        Predict recall probability after delta_t days.
        
        Args:
            user_id: Unique identifier for the user
            item_id: Unique identifier for the item
            delta_t: Days since last review
            
        Returns:
            Predicted recall probability (0-1)
        """
        pass
    
    @abstractmethod
    def calculate_half_life(self, user_id: str, item_id: str) -> float:
        """
        Calculate memory half-life for an item.
        
        Args:
            user_id: Unique identifier for the user
            item_id: Unique identifier for the item
            
        Returns:
            Half-life in days
        """
        pass
    
    def get_state(self, user_id: str, item_id: str) -> SchedulerState:
        """Get the current state for a user-item pair."""
        key = (user_id, item_id)
        if key not in self.states:
            self.states[key] = SchedulerState()
        return self.states[key]
    
    def update_state(self, user_id: str, item_id: str, state: SchedulerState) -> None:
        """Update the state for a user-item pair."""
        self.states[(user_id, item_id)] = state
    
    def get_statistics(self) -> Dict[str, int]:
        """Get scheduler statistics."""
        return {
            'total_reviews': self.total_reviews,
            'total_items': len(self.states),
            'active_items': sum(1 for state in self.states.values() if state.repetitions > 0)
        }
    
    def reset(self) -> None:
        """Reset the scheduler state."""
        self.states.clear()
        self.total_reviews = 0
        self.total_items = 0
    
    def get_all_half_lives(self) -> Dict[Tuple[str, str], float]:
        """
        Get half-lives for all tracked items.
        
        Returns:
            dict: (user_id, item_id) -> half_life mapping
        """
        half_lives = {}
        for (user_id, item_id), state in self.states.items():
            half_life = self.calculate_half_life(user_id, item_id)
            half_lives[(user_id, item_id)] = half_life
        return half_lives
    
    def get_all_recall_predictions(self, delta_t: float) -> Dict[Tuple[str, str], float]:
        """
        Get recall predictions for all tracked items at delta_t days.
        
        Args:
            delta_t: Days since last review
            
        Returns:
            dict: (user_id, item_id) -> recall_probability mapping
        """
        predictions = {}
        for (user_id, item_id), state in self.states.items():
            p_recall = self.predict_recall(user_id, item_id, delta_t)
            predictions[(user_id, item_id)] = p_recall
        return predictions