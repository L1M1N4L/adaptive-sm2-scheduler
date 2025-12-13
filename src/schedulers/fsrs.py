"""
FSRS (Free Spaced Repetition Scheduler) Implementation

This module implements the FSRS algorithm, adapted from the DHP (Difficulty, Halflife, Probability)
model found in evaluation/SSP-MMC-Plus/model/DHP.py.
"""

import math
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from .base import BaseScheduler, ScheduleDecision, SchedulerState, ReviewRecord

class FSRSScheduler(BaseScheduler):
    """
    FSRS Scheduler implementation.
    
    References:
        - Adapted from DHP model: evaluation/SSP-MMC-Plus/model/DHP.py
    """
    
    def __init__(self, use_default_params: bool = True):
        super().__init__()
        
        # Default FSRS/DHP parameters
        # These correspond to ra, rb, rc, rd, fa, fb, fc, fd in DHP.py
        # Since we don't have a trained model, we use reasonable defaults
        # that provide a working scheduler.
        
        # Recall Halflife Params (ra, rb, rc, rd)
        self.ra = 0.5   # Base increase
        self.rb = 0.1   # Effect of difficulty
        self.rc = 0.1   # Effect of current halflife
        self.rd = 0.5   # Effect of retrievability (1-p)
        
        # Forget Halflife Params (fa, fb, fc, fd)
        self.fa = 0.1   # Base decrease
        self.fb = 0.1   # Effect of difficulty
        self.fc = 0.1   # Effect of current halflife
        self.fd = 0.1   # Effect of retrievability
        
        # Default starting difficulty
        self.default_difficulty = 5.0
        
    def _cal_start_halflife(self, d: float) -> float:
        """Calculate initial halflife based on difficulty."""
        # Formula from DHP.py: - 1 / np.log2(max(0.925 - 0.05 * d, 0.025))
        return -1.0 / math.log2(max(0.925 - 0.05 * d, 0.025))
        
    def _cal_recall_halflife(self, d: float, halflife: float, p_recall: float) -> float:
        """Calculate new halflife after successful recall."""
        # Formula from DHP.py
        # h * (1 + exp(ra) * d^rb * h^rc * (1-p)^rd)
        term = (
            math.exp(self.ra) * 
            math.pow(d, self.rb) * 
            math.pow(halflife, self.rc) * 
            math.pow(1 - p_recall, self.rd)
        )
        return halflife * (1 + term)
        
    def _cal_forget_halflife(self, d: float, halflife: float, p_recall: float) -> float:
        """Calculate new halflife after failed recall (forgetting)."""
        # Formula from DHP.py
        # exp(fa) * d^fb * h^fc * (1-p)^fd
        return (
            math.exp(self.fa) * 
            math.pow(d, self.fb) * 
            math.pow(halflife, self.fc) * 
            math.pow(1 - p_recall, self.fd)
        )

    def schedule_review(self, user_id: str, item_id: str, rating: int, timestamp: float) -> ScheduleDecision:
        """
        Schedule review using FSRS/DHP logic.
        
        We store 'difficulty' (d) in the ease_factor field of SchedulerState
        to avoid modifying the base class.
        """
        state = self.get_state(user_id, item_id)
        
        # Initialize difficulty if this is a first review
        # Note: In standard SM-2 ease_factor defaults to 2.5
        # If this looks like default SM-2, we treat it as uninitialized for FSRS
        if state.repetitions == 0:
            difficulty = self.default_difficulty
        else:
            # We store 'd' in ease_factor
            difficulty = state.ease_factor
            
        # Determine recall result (binary in DHP: 1=recall, 0=forget)
        # Mapping 0-5 rating to binary:
        # 0 (Again), 1, 2 -> Forget
        # 3 (Hard), 4 (Good), 5 (Easy) -> Recall
        is_recall = rating >= 3
        
        # Calculate time since last review (delta_t)
        if state.last_review == 0:
            delta_t = 0
        else:
            delta_t = timestamp - state.last_review
            
        # Current halflife (stored in interval)
        current_halflife = float(state.interval) if state.interval > 0 else 0.0
        
        next_halflife = 0.0
        next_difficulty = difficulty
        
        if state.repetitions == 0:
            # First review ever
            next_halflife = self._cal_start_halflife(difficulty)
            # Adjust difficulty based on first rating?
            # DHP doesn't show explicit first-rating adjustment, but let's do a simple one
            # to separate easy vs hard items immediately
            if rating < 3:
                next_difficulty = min(difficulty + 2, 18)
            elif rating == 5:
                next_difficulty = max(difficulty - 2, 1)
                
        else:
            # Subsequent reviews
            
            # Predict P(recall) at this moment
            # p_recall = 2^(-delta_t / h)
            if current_halflife <= 0:
                p_recall = 0
            else:
                p_recall = 2 ** (-delta_t / current_halflife)
                
            if is_recall:
                next_halflife = self._cal_recall_halflife(difficulty, current_halflife, p_recall)
                # DHP updates difficulty only on forget? 
                # "if recall == 1 ... else ... d = min(d + 2, 18)"
                # But intuitively difficulty should decrease (or stay same) on success
                # Standard FSRS adapts difficulty on every review.
                # Let's keep DHP logic strictly: it only modifies d on forget in the snippet provided.
                # However, to be a good scheduler, we probably want d to drop if rating is 5
                if rating == 5:
                     next_difficulty = max(difficulty - 1, 1) # Bonus logic for "Easy"
            else:
                next_halflife = self._cal_forget_halflife(difficulty, current_halflife, p_recall)
                # DHP logic for forget: d = min(d + 2, 18)
                next_difficulty = min(difficulty + 2, 18)
        
        # Update state
        # In FSRS, interval is often defined as when Retrieval Probability (R) falls to target (e.g. 0.9)
        # R = 2^(-t/h)  =>  0.9 = 2^(-t/h) => t = -h * log2(0.9)
        # NOTE: DHP.py seems to treat 'h' as the interval itself roughly (halflife)
        # If we schedule exactly at halflife, R=0.5, which is too hard.
        # We usually want R=0.9 or so.
        
        target_retention = 0.9
        next_interval = max(1, int(-next_halflife * math.log2(target_retention)))
        
        # Save state
        state.interval = max(1, int(next_halflife)) # Wait, let's store halflife in interval?
        # NO, 'interval' in BaseScheduler is int (days until next review).
        # We need to store exact halflife somewhere.
        # BaseScheduler state is rigid.
        # Let's store halflife in 'interval' for now as it's the closest concept,
        # BUT 'interval' is integer.
        # This might lose precision.
        # However, for a demo app, int(halflife) might be precise enough if halflife is days.
        # Actually, let's use the calculated next_interval for the UI (the ScheduleDecision),
        # but we need to PERIST the halflife for the next calculation.
        # Use state.interval to store the 'integer interval' for scheduling,
        # and maybe encoded in ease_factor? No, ease_factor is float.
        # Let's stick to using state.interval as the halflife approx.
        # (Or we could abuse repetitions? No)
        
        # REVISION: DHP says "interval" in line 96.
        # The code calculates 'h'.
        # Let's assume for this implementation that state.interval stores the HALFLIFE.
        # And the returned decision.interval is the recommended spacing days.
        
        state.interval = int(next_halflife)
        if state.interval < 1: state.interval = 1
        
        state.ease_factor = next_difficulty
        state.repetitions += 1
        state.last_review = timestamp
        
        self.update_state(user_id, item_id, state)
        self.total_reviews += 1
        
        # Calculate P(recall) for the decision display
        # (This is P(recall) *now*, before the update? Or predicted at the *next* interval?)
        # Decision docstring says "Predicted recall probability". Usually at the end of the interval.
        predicted_recall_at_interval = 2 ** (-next_interval / next_halflife) if next_halflife > 0 else 0
        
        return ScheduleDecision(
            interval=next_interval,
            ease_factor=next_difficulty,
            repetitions=state.repetitions,
            p_recall=predicted_recall_at_interval,
            confidence=1.0 # Static confidence
        )

    def predict_recall(self, user_id: str, item_id: str, delta_t: float) -> float:
        """Predict recall probability."""
        state = self.get_state(user_id, item_id)
        halflife = float(state.interval) if state.interval > 0 else 1.0
        return 2 ** (-delta_t / halflife)

    def calculate_half_life(self, user_id: str, item_id: str) -> float:
        """Calculate half-life (just return stored halflife)."""
        state = self.get_state(user_id, item_id)
        return float(state.interval)
