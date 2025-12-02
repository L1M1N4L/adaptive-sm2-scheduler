"""
Hybrid Spaced Repetition Scheduler

Combines SM-2 algorithm with ML-based half-life prediction models (HLR, DHP, RNN_HLR)
for adaptive spaced repetition scheduling.

The hybrid approach:
- Uses SM-2 for new items and as a fallback
- Gradually incorporates ML predictions as review history accumulates
- Adaptively blends SM-2 and ML predictions based on confidence and experience
"""

import math
import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

# Add evaluation model directory to path
_eval_model_path = Path(__file__).parent.parent.parent / "evaluation" / "SSP-MMC-Plus" / "model"
if str(_eval_model_path) not in sys.path:
    sys.path.insert(0, str(_eval_model_path))

from .base import (
    BaseScheduler,
    SchedulerState,
    ScheduleDecision,
    ReviewRecord,
    exponential_forgetting_curve,
    calculate_half_life_from_recall
)
from .sm2 import SM2Scheduler

# Import ML models
try:
    from halflife_regression import SpacedRepetitionModel as HLRModel, Instance, pclip, hclip, MIN_HALF_LIFE, MAX_HALF_LIFE
    HLR_AVAILABLE = True
except ImportError:
    HLR_AVAILABLE = False
    print("Warning: HLR model not available")

try:
    # DHP imports from model.utils, so we need to import utils first
    import importlib.util
    utils_path = _eval_model_path / "utils.py"
    dhp_path = _eval_model_path / "DHP.py"
    
    if utils_path.exists() and dhp_path.exists():
        # Import utils first
        utils_spec = importlib.util.spec_from_file_location("model.utils", utils_path)
        utils_module = importlib.util.module_from_spec(utils_spec)
        # Add to sys.modules so DHP can import it
        sys.modules['model.utils'] = utils_module
        sys.modules['model'] = type(sys)('model')
        sys.modules['model'].utils = utils_module
        utils_spec.loader.exec_module(utils_module)
        lineToTensor = utils_module.lineToTensor
        
        # Now import DHP
        dhp_spec = importlib.util.spec_from_file_location("model.DHP", dhp_path)
        dhp_module = importlib.util.module_from_spec(dhp_spec)
        dhp_spec.loader.exec_module(dhp_module)
        DHPModel = dhp_module.SpacedRepetitionModel
        DHP_AVAILABLE = True
    else:
        DHP_AVAILABLE = False
except Exception as e:
    DHP_AVAILABLE = False
    print(f"Warning: DHP model not available: {e}")

try:
    import torch
    RNN_AVAILABLE = False  # Disable RNN by default due to dependency issues
    # RNN requires trained models and has complex dependencies
    # Uncomment below if you have a trained RNN model and want to use it
    # try:
    #     import importlib.util
    #     rnn_path = _eval_model_path / "RNN_HLR.py"
    #     if rnn_path.exists():
    #         spec = importlib.util.spec_from_file_location("RNN_HLR", rnn_path)
    #         rnn_module = importlib.util.module_from_spec(spec)
    #         spec.loader.exec_module(rnn_module)
    #         RNNModel = rnn_module.SpacedRepetitionModel
    #         RNN_AVAILABLE = True
    #     else:
    #         RNN_AVAILABLE = False
    # except Exception as e:
    #     RNN_AVAILABLE = False
    #     print(f"Warning: RNN model not available: {e}")
except ImportError:
    RNN_AVAILABLE = False
    # print("Warning: RNN model not available (PyTorch required)")


class HybridScheduler(BaseScheduler):
    """
    Hybrid scheduler combining SM-2 with ML-based half-life prediction.
    
    This scheduler uses an adaptive blending strategy:
    - Beta (β) controls the blend: interval = (1-β) * SM2_interval + β * ML_interval
    - Beta starts low (0.0-0.2) for new items, increases with review history
    - Falls back to pure SM-2 if ML models fail or aren't available
    """
    
    # SM-2 constants (inherited from SM2Scheduler)
    DEFAULT_EF = 2.5
    MIN_EF = 1.3
    FIRST_INTERVAL = 1
    SECOND_INTERVAL = 6
    
    # Hybrid parameters
    MIN_BETA = 0.0  # Minimum ML weight (start with pure SM-2)
    MAX_BETA = 0.8  # Maximum ML weight (never fully replace SM-2)
    BETA_GROWTH_RATE = 0.1  # Beta increase per review after threshold
    BETA_THRESHOLD = 3  # Reviews needed before beta starts increasing
    
    def __init__(self, 
                 use_hlr: bool = True,
                 use_dhp: bool = True,
                 use_rnn: bool = False,
                 hlr_weights_path: Optional[str] = None,
                 dhp_params_path: Optional[str] = None,
                 rnn_model_path: Optional[str] = None,
                 pure_ml: bool = False):
        """
        Initialize hybrid scheduler.
        
        Args:
            use_hlr: Use Half-Life Regression model
            use_dhp: Use Difficulty-Half-life-Prediction model
            use_rnn: Use RNN-based half-life prediction (requires PyTorch)
            hlr_weights_path: Path to pre-trained HLR weights file
            dhp_params_path: Path to pre-trained DHP parameters file
            rnn_model_path: Path to pre-trained RNN model file
            pure_ml: If True, use only ML predictions (beta=1.0), no SM-2 blending
        """
        super().__init__()
        self.sm2_scheduler = SM2Scheduler()
        self.pure_ml = pure_ml
        
        # ML model flags
        self.use_hlr = use_hlr and HLR_AVAILABLE
        self.use_dhp = use_dhp and DHP_AVAILABLE
        self.use_rnn = use_rnn and RNN_AVAILABLE
        
        # Initialize ML models
        self.hlr_model = None
        self.dhp_model = None
        self.rnn_model = None
        
        # Initialize HLR model
        if self.use_hlr:
            # Create dummy trainset to avoid division by zero
            # Instance(p, t, fv, h, a, right, wrong, spelling)
            dummy_instance = Instance(
                p=0.5, 
                t=1, 
                fv=[('right', 1.0), ('wrong', 0.0), ('bias', 1.0)], 
                h=1.0, 
                a=0.5,
                right=1,
                wrong=0,
                spelling='dummy'
            )
            self.hlr_model = HLRModel(trainset=[dummy_instance], testset=[dummy_instance], method='HLR', omit_h_term=False, omit_lexemes=True)
            if hlr_weights_path and os.path.exists(hlr_weights_path):
                self._load_hlr_weights(hlr_weights_path)
        
        # Initialize DHP model
        if self.use_dhp:
            self.dhp_model = DHPModel(train_set=None, test_set=None)
            if dhp_params_path and os.path.exists(dhp_params_path):
                self._load_dhp_params(dhp_params_path)
        
        # Initialize RNN model
        if self.use_rnn:
            if rnn_model_path and os.path.exists(rnn_model_path):
                try:
                    self.rnn_model = torch.jit.load(rnn_model_path)
                    self.rnn_model.eval()
                except Exception as e:
                    print(f"Warning: Could not load RNN model: {e}")
                    self.use_rnn = False
    
    def _load_hlr_weights(self, weights_path: str):
        """Load HLR model weights from file."""
        try:
            with open(weights_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        feature, weight = parts[0], float(parts[1])
                        self.hlr_model.weights[feature] = weight
        except Exception as e:
            print(f"Warning: Could not load HLR weights: {e}")
    
    def _load_dhp_params(self, params_path: str):
        """Load DHP model parameters from CSV file."""
        try:
            import pandas as pd
            params_df = pd.read_csv(params_path)
            if not params_df.empty:
                self.dhp_model.ra = params_df['ra'].iloc[0]
                self.dhp_model.rb = params_df['rb'].iloc[0]
                self.dhp_model.rc = params_df['rc'].iloc[0]
                self.dhp_model.rd = params_df['rd'].iloc[0]
                self.dhp_model.fa = params_df['fa'].iloc[0]
                self.dhp_model.fb = params_df['fb'].iloc[0]
                self.dhp_model.fc = params_df['fc'].iloc[0]
                self.dhp_model.fd = params_df['fd'].iloc[0]
        except Exception as e:
            print(f"Warning: Could not load DHP parameters: {e}")
    
    def schedule_review(self,
                       user_id: str,
                       item_id: str,
                       rating: int,
                       timestamp: float) -> ScheduleDecision:
        """
        Schedule next review using hybrid SM-2 + ML approach.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            rating: Quality score 0-5
            timestamp: Current timestamp (days)
            
        Returns:
            ScheduleDecision with blended interval and predictions
        """
        # Get current state
        state = self.get_state(user_id, item_id)
        
        # Get SM-2 prediction
        sm2_decision = self.sm2_scheduler.schedule_review(
            user_id, item_id, rating, timestamp
        )
        
        # Calculate adaptive beta based on review history
        beta = self._calculate_adaptive_beta(state)
        
        # Get ML prediction if available
        ml_interval = None
        ml_half_life = None
        ml_confidence = 0.0
        
        # For pure ML, try ML prediction even with fewer reviews (lower threshold)
        # For hybrid, use normal threshold
        ml_threshold_reviews = 1 if self.pure_ml else self.BETA_THRESHOLD
        
        if beta > 0 and state.repetitions >= ml_threshold_reviews:
            ml_prediction = self._predict_with_ml(user_id, item_id, state, rating, timestamp)
            if ml_prediction:
                ml_interval, ml_half_life, ml_confidence = ml_prediction
            else:
                # Debug: check why ML prediction failed
                # This happens when models return None
                ml_interval = None
                ml_half_life = None
                ml_confidence = 0.0
        
        # Blend SM-2 and ML predictions
        # For pure ML mode, use lower confidence threshold to allow ML to contribute
        # even with untrained models. For hybrid, use slightly higher threshold.
        ml_threshold = 0.2 if self.pure_ml else 0.1
        
        if ml_interval is not None and ml_confidence > ml_threshold:
            # Use blended interval (or pure ML if beta=1.0)
            blended_interval = int(
                (1 - beta) * sm2_decision.interval + beta * ml_interval
            )
            blended_half_life = (
                (1 - beta) * self._sm2_half_life(sm2_decision) + 
                beta * ml_half_life
            )
            final_confidence = beta * ml_confidence
        else:
            # Fall back to SM-2 (unless pure ML mode)
            if self.pure_ml:
                # For pure ML, we MUST use ML predictions - if they're not available,
                # try to generate a default ML prediction based on current state
                if ml_interval is None:
                    # Generate a simple ML-based interval from half-life
                    # Use a conservative estimate based on current state
                    if state.repetitions > 0:
                        # Estimate half-life from review history
                        avg_rating = sum(r.rating for r in state.review_history[-5:]) / len(state.review_history[-5:]) if len(state.review_history) >= 5 else 3.0
                        estimated_h = max(1.0, state.interval * (1.5 if avg_rating >= 3 else 0.7))
                        ml_interval = max(1, int(estimated_h / 1.5))
                        ml_half_life = estimated_h
                        ml_confidence = 0.25  # Low but non-zero confidence
                    else:
                        # First review - use default
                        ml_interval = 1
                        ml_half_life = 1.0
                        ml_confidence = 0.2
                
                # Use ML prediction (even if low confidence)
                blended_interval = ml_interval
                blended_half_life = ml_half_life
                final_confidence = ml_confidence
            else:
                blended_interval = sm2_decision.interval
                blended_half_life = self._sm2_half_life(sm2_decision)
                final_confidence = 0.0
                beta = 0.0  # No ML contribution
        
        # Predict recall probability
        p_recall = exponential_forgetting_curve(blended_interval, blended_half_life)
        
        # Update state (use SM-2 state as base, but with blended interval)
        state.ease_factor = sm2_decision.ease_factor
        state.repetitions = sm2_decision.repetitions
        state.interval = blended_interval
        state.last_review = timestamp
        
        # Add to review history
        review_record = ReviewRecord(
            user_id=user_id,
            item_id=item_id,
            rating=rating,
            timestamp=timestamp,
            interval=blended_interval,
            ease_factor=sm2_decision.ease_factor,
            repetitions=sm2_decision.repetitions
        )
        state.review_history.append(review_record)
        
        # Update scheduler statistics
        self.total_reviews += 1
        if (user_id, item_id) not in self.states:
            self.total_items += 1
        
        # Update state in scheduler
        self.update_state(user_id, item_id, state)
        
        return ScheduleDecision(
            interval=blended_interval,
            ease_factor=sm2_decision.ease_factor,
            repetitions=sm2_decision.repetitions,
            p_recall=p_recall,
            confidence=final_confidence
        )
    
    def predict_recall(self, user_id: str, item_id: str, delta_t: float) -> float:
        """
        Predict recall probability after delta_t days using hybrid approach.
        
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
        
        # Get SM-2 prediction
        sm2_half_life = self.sm2_scheduler.calculate_half_life(user_id, item_id)
        sm2_p_recall = exponential_forgetting_curve(delta_t, sm2_half_life)
        
        # Get ML prediction if available
        beta = self._calculate_adaptive_beta(state)
        if beta > 0 and state.repetitions >= self.BETA_THRESHOLD:
            ml_prediction = self._predict_with_ml(user_id, item_id, state, None, None)
            if ml_prediction:
                _, ml_half_life, ml_confidence = ml_prediction
                if ml_confidence > 0.3:
                    ml_p_recall = exponential_forgetting_curve(delta_t, ml_half_life)
                    # Blend predictions
                    return (1 - beta) * sm2_p_recall + beta * ml_p_recall
        
        return sm2_p_recall
    
    def calculate_half_life(self, user_id: str, item_id: str) -> float:
        """
        Calculate memory half-life using hybrid approach.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            
        Returns:
            Half-life in days
        """
        state = self.get_state(user_id, item_id)
        if state.repetitions == 0:
            return 1.0
        
        # Get SM-2 half-life
        sm2_half_life = self.sm2_scheduler.calculate_half_life(user_id, item_id)
        
        # Get ML prediction if available
        beta = self._calculate_adaptive_beta(state)
        if beta > 0 and state.repetitions >= self.BETA_THRESHOLD:
            ml_prediction = self._predict_with_ml(user_id, item_id, state, None, None)
            if ml_prediction:
                _, ml_half_life, ml_confidence = ml_prediction
                if ml_confidence > 0.3:
                    # Blend half-lives
                    return (1 - beta) * sm2_half_life + beta * ml_half_life
        
        return sm2_half_life
    
    def _calculate_adaptive_beta(self, state: SchedulerState) -> float:
        """
        Calculate adaptive beta (ML weight) for blending.
        
        If pure_ml=True, always returns 1.0 (100% ML, 0% SM-2).
        Otherwise, uses adaptive strategy based on review history.
        
        Beta increases with:
        - Number of reviews (more data = more confidence in ML)
        - Review consistency (lower variance = higher confidence)
        
        Args:
            state: Current scheduler state
            
        Returns:
            Beta value between MIN_BETA and MAX_BETA (or 1.0 for pure ML mode)
        """
        # Pure ML mode: always use ML predictions
        if self.pure_ml:
            return 1.0
        
        if state.repetitions < self.BETA_THRESHOLD:
            return self.MIN_BETA
        
        # Base growth: more reviews = higher beta
        reviews_beyond_threshold = state.repetitions - self.BETA_THRESHOLD
        base_beta = min(
            self.MIN_BETA + reviews_beyond_threshold * self.BETA_GROWTH_RATE,
            self.MAX_BETA
        )
        
        # Adjust based on review history consistency
        if len(state.review_history) >= 3:
            recent_ratings = [r.rating for r in state.review_history[-5:]]
            if recent_ratings:
                avg_rating = sum(recent_ratings) / len(recent_ratings)
                # Higher average rating = more consistent = higher confidence
                consistency_factor = (avg_rating - 2) / 3.0  # Normalize to 0-1
                base_beta *= (0.7 + 0.3 * consistency_factor)
        
        return min(base_beta, self.MAX_BETA)
    
    def _predict_with_ml(self,
                        user_id: str,
                        item_id: str,
                        state: SchedulerState,
                        rating: Optional[int],
                        timestamp: Optional[float]) -> Optional[Tuple[int, float, float]]:
        """
        Predict interval and half-life using ML models.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            state: Current scheduler state
            rating: Current rating (if available)
            timestamp: Current timestamp (if available)
            
        Returns:
            Tuple of (interval, half_life, confidence) or None if prediction fails
        """
        predictions = []
        confidences = []
        
        # Try HLR model
        if self.use_hlr and self.hlr_model:
            hlr_pred = self._predict_hlr(user_id, item_id, state, rating, timestamp)
            if hlr_pred:
                predictions.append(hlr_pred)
                # Adjust confidence based on whether model has trained weights
                has_weights = len(self.hlr_model.weights) > 3  # More than just defaults
                # For pure ML, use higher base confidence to allow it to work
                # For hybrid, use lower confidence for untrained models
                if self.pure_ml:
                    base_confidence = 0.5 if has_weights else 0.3
                else:
                    base_confidence = 0.6 if has_weights else 0.3
                confidences.append(base_confidence)
        
        # Try DHP model
        if self.use_dhp and self.dhp_model:
            dhp_pred = self._predict_dhp(user_id, item_id, state, rating, timestamp)
            if dhp_pred:
                predictions.append(dhp_pred)
                # Adjust confidence based on whether model has trained parameters
                has_params = (abs(self.dhp_model.ra) > 0.001 or 
                             abs(self.dhp_model.fa) > 0.001)  # Check if trained
                # For pure ML, use higher base confidence to allow it to work
                if self.pure_ml:
                    base_confidence = 0.6 if has_params else 0.4
                else:
                    base_confidence = 0.7 if has_params else 0.4
                confidences.append(base_confidence)
        
        # Try RNN model (if enabled and available)
        if self.use_rnn and self.rnn_model:
            rnn_pred = self._predict_rnn(user_id, item_id, state, rating, timestamp)
            if rnn_pred:
                predictions.append(rnn_pred)
                confidences.append(0.8)  # Highest confidence for RNN
        
        if not predictions:
            return None
        
        # Ensemble: weighted average of available predictions
        if len(predictions) == 1:
            interval, half_life = predictions[0]
            confidence = confidences[0]
        else:
            # Weight by confidence
            total_weight = sum(confidences)
            weighted_interval = sum(p[0] * c for p, c in zip(predictions, confidences)) / total_weight
            weighted_half_life = sum(p[1] * c for p, c in zip(predictions, confidences)) / total_weight
            interval = int(weighted_interval)
            half_life = weighted_half_life
            confidence = sum(confidences) / len(confidences)  # Average confidence
        
        return (interval, half_life, confidence)
    
    def _predict_hlr(self,
                    user_id: str,
                    item_id: str,
                    state: SchedulerState,
                    rating: Optional[int],
                    timestamp: Optional[float]) -> Optional[Tuple[int, float]]:
        """
        Predict using Half-Life Regression model.
        """
        try:
            # For pure ML, allow predictions even with no history (use defaults)
            # For hybrid, require at least 1 review
            min_history = 0 if self.pure_ml else 1
            if len(state.review_history) < min_history:
                # For pure ML with no history, use default predictions
                if self.pure_ml and len(state.review_history) == 0:
                    # Use default half-life and interval for new items
                    default_h = 1.0
                    default_interval = 1
                    return (default_interval, default_h)
                return None
            
            # Extract features from review history
            right_count = sum(1 for r in state.review_history if r.rating >= 3)
            wrong_count = len(state.review_history) - right_count
            
            # Create feature vector: sqrt(1 + right), sqrt(1 + wrong), bias
            fv = [
                (sys.intern('right'), math.sqrt(1 + right_count)),
                (sys.intern('wrong'), math.sqrt(1 + wrong_count)),
                (sys.intern('bias'), 1.0)
            ]
            
            # Get current interval (time since last review)
            if timestamp is not None and state.last_review > 0:
                t = max(1, int(timestamp - state.last_review))
            else:
                t = state.interval if state.interval > 0 else 1
            
            # Create Instance for HLR model
            # Instance(p, t, fv, h, a, right, wrong, spelling) where:
            # p = recall probability (we'll use predicted)
            # t = time interval
            # fv = feature vector
            # h = half-life (we'll get from prediction)
            # a = accuracy estimate
            # right = right count
            # wrong = wrong count
            # spelling = item identifier (optional)
            p_estimate = (right_count + 2.0) / (right_count + wrong_count + 4.0) if (right_count + wrong_count) > 0 else 0.5
            h_estimate = -t / math.log2(p_estimate) if p_estimate > 0 else 1.0
            
            inst = Instance(
                p=pclip(p_estimate), 
                t=t, 
                fv=fv, 
                h=hclip(h_estimate), 
                a=p_estimate,
                right=right_count,
                wrong=wrong_count,
                spelling=item_id
            )
            
            # Predict using HLR model
            p_pred, h_pred = self.hlr_model.predict(inst, base=2.0)
            
            # Validate prediction
            if h_pred is None or h_pred <= 0 or not math.isfinite(h_pred):
                return None
            
            # Convert half-life to interval (interval ≈ half_life / 1.5 for ~90% recall)
            interval = max(1, int(h_pred / 1.5))
            
            # Ensure reasonable values
            if interval < 1 or h_pred < 0.01:
                return None
            
            return (interval, h_pred)
        except Exception as e:
            # Fall back silently - uncomment for debugging
            # print(f"HLR prediction error: {e}")
            return None
    
    def _predict_dhp(self,
                    user_id: str,
                    item_id: str,
                    state: SchedulerState,
                    rating: Optional[int],
                    timestamp: Optional[float]) -> Optional[Tuple[int, float]]:
        """
        Predict using Difficulty-Half-life-Prediction model.
        """
        try:
            # For pure ML, allow predictions with 0 reviews (use defaults)
            # For hybrid, require at least 2 reviews for DHP
            min_history = 0 if self.pure_ml else 2
            if len(state.review_history) < min_history:
                # For pure ML with no/minimal history, use default predictions
                if self.pure_ml and len(state.review_history) == 0:
                    # Use default difficulty and half-life
                    default_d = 1
                    default_h = self.dhp_model.cal_start_halflife(default_d) if hasattr(self.dhp_model, 'cal_start_halflife') else 1.0
                    default_interval = max(1, int(default_h / 1.5))
                    return (default_interval, default_h)
                return None
            
            # Calculate difficulty (d) from review history
            wrong_count = sum(1 for r in state.review_history if r.rating < 3)
            d = min(wrong_count + 1, 18)  # Cap at 18
            
            # Build review history in DHP format: (recall, interval)
            # DHP expects: recall=1 for correct, recall=0 for wrong
            r_history_str = ','.join(['1' if r.rating >= 3 else '0' for r in state.review_history])
            t_history_str = ','.join([str(int(r.interval)) if r.interval > 0 else '1' for r in state.review_history])
            
            # Calculate p_history (predicted recall probabilities)
            p_history_list = []
            current_h = 1.0
            for i, r in enumerate(state.review_history):
                if i == 0:
                    # First review: use start half-life
                    current_h = self.dhp_model.cal_start_halflife(d)
                    p_history_list.append(str(2.0 ** (-r.interval / current_h) if r.interval > 0 else 0.5))
                else:
                    # Subsequent reviews: use DHP update
                    recall = 1 if r.rating >= 3 else 0
                    interval = int(r.interval) if r.interval > 0 else 1
                    prev_interval = int(state.review_history[i-1].interval) if i > 0 and state.review_history[i-1].interval > 0 else 1
                    
                    if recall == 1:
                        p_recall = 2.0 ** (-prev_interval / current_h)
                        current_h = self.dhp_model.cal_recall_halflife(d, current_h, p_recall)
                    else:
                        p_recall = 2.0 ** (-prev_interval / current_h)
                        current_h = self.dhp_model.cal_forget_halflife(d, current_h, p_recall)
                        d = min(d + 2, 18)
                    
                    p_history_list.append(str(p_recall))
            
            p_history_str = ','.join(p_history_list)
            
            # Create line tensor for DHP
            line = (r_history_str, t_history_str, p_history_str)
            line_tensor = lineToTensor([line])[0]
            
            # Process through DHP model
            ph = 0.0
            d_current = d
            for j in range(line_tensor.size()[0]):
                ph, d_current = self.dhp_model.dhp(line_tensor[j][0], ph, d_current)
            
            # Validate prediction
            if ph is None or ph <= 0 or not math.isfinite(ph):
                return None
            
            # Convert half-life to interval
            interval = max(1, int(ph / 1.5))
            
            # Ensure reasonable values
            if interval < 1 or ph < 0.01:
                return None
            
            return (interval, ph)
        except Exception as e:
            # Fall back silently - uncomment for debugging
            # print(f"DHP prediction error: {e}")
            return None
    
    def _predict_rnn(self,
                    user_id: str,
                    item_id: str,
                    state: SchedulerState,
                    rating: Optional[int],
                    timestamp: Optional[float]) -> Optional[Tuple[int, float]]:
        """
        Predict using RNN-based half-life regression.
        """
        try:
            if len(state.review_history) < 1:
                return None
            
            # Build review history strings
            r_history_str = ','.join(['1' if r.rating >= 3 else '0' for r in state.review_history])
            t_history_str = ','.join([str(int(r.interval)) if r.interval > 0 else '1' for r in state.review_history])
            
            # Calculate p_history (predicted recall probabilities)
            p_history_list = []
            for r in state.review_history:
                # Estimate recall probability based on rating
                if r.rating >= 4:
                    p = 0.9
                elif r.rating == 3:
                    p = 0.7
                else:
                    p = 0.3
                p_history_list.append(str(p))
            
            p_history_str = ','.join(p_history_list)
            
            # Create tensor
            line = (r_history_str, t_history_str, p_history_str)
            line_tensor = lineToTensor([line])[0]
            
            # Run RNN inference
            with torch.no_grad():
                output = self.rnn_model(line_tensor, None)
                half_life = float(output[0][0][0])
            
            # Convert half-life to interval
            interval = max(1, int(half_life / 1.5))
            
            return (interval, half_life)
        except Exception as e:
            return None
    
    def _sm2_half_life(self, decision: ScheduleDecision) -> float:
        """Calculate half-life from SM-2 decision."""
        return max(decision.interval / 1.5, 1.0)
    
    def reset(self) -> None:
        """Reset scheduler state."""
        super().reset()
        self.sm2_scheduler.reset()
