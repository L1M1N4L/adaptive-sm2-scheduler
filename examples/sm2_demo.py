"""
SM-2 Scheduler Demo for Braille Learning

This demo shows how to use the SM-2 scheduler for Braille character learning.
Perfect for testing with the FSRS-Anki-20k dataset when it's downloaded.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.schedulers import SM2Scheduler, SM2StandaloneCard, RatingConverter
import numpy as np


def simple_usage_demo():
    """Show basic usage of SM-2 scheduler."""
    print("=== SM-2 Scheduler Basic Usage ===\n")
    
    # Method 1: Using the integrated scheduler (recommended)
    print("--- Method 1: Integrated Scheduler ---")
    scheduler = SM2Scheduler()
    
    user_id = "learner_001"
    item_id = "braille_a"
    
    # Simulate learning a Braille character
    reviews = [
        (0, 4),   # Day 0: Good recall
        (1, 5),   # Day 1: Perfect recall  
        (7, 3),   # Day 7: Difficult recall
        (15, 4),  # Day 15: Good recall
    ]
    
    print(f"Learning '{item_id}' for user '{user_id}':")
    for day, rating in reviews:
        decision = scheduler.schedule_review(
            user_id=user_id,
            item_id=item_id,
            rating=rating,
            timestamp=day
        )
        
        state = scheduler.get_state(user_id, item_id)
        params = state.algorithm_params
        
        print(f"  Day {day:2d} | Rating={rating} | Next: {decision.interval:.0f} days | "
              f"EF={params['ease_factor']:.2f} | p_recall={decision.p_recall:.3f}")
    
    # Method 2: Using standalone card (simpler, backwards compatible)
    print("\n--- Method 2: Standalone Card ---")
    card = SM2StandaloneCard()
    
    for rating in [4, 5, 3, 4]:
        interval = card.review(rating)
        state = card.get_state()
        print(f"  Rating={rating} → {interval} days | {card}")


def braille_learning_scenario():
    """Complete Braille learning scenario."""
    print("\n=== Braille Learning Scenario ===\n")
    
    scheduler = SM2Scheduler()
    user_id = "braille_learner"
    
    # Braille alphabet (first 5 letters)
    braille_chars = {
        "braille_a": "⠁",  # A
        "braille_b": "⠃",  # B  
        "braille_c": "⠉",  # C
        "braille_d": "⠙",  # D
        "braille_e": "⠑",  # E
    }
    
    print("Learning Braille alphabet with SM-2:")
    print("(Simulating realistic learning patterns)\n")
    
    for char_id, symbol in braille_chars.items():
        print(f"--- Learning '{char_id}' ({symbol}) ---")
        
        # Simulate different learning patterns
        if char_id == "braille_a":
            # Easy learner - mostly good ratings
            reviews = [(0, 4), (1, 5), (7, 4), (15, 5), (30, 5)]
        elif char_id == "braille_b":
            # Struggling learner - some difficulties
            reviews = [(0, 3), (1, 2), (2, 3), (3, 4), (10, 3), (20, 4)]
        elif char_id == "braille_c":
            # Perfect learner - all excellent
            reviews = [(0, 5), (1, 5), (7, 5), (15, 5)]
        elif char_id == "braille_d":
            # Inconsistent learner - mixed results
            reviews = [(0, 4), (1, 2), (2, 4), (3, 3), (10, 4), (20, 5)]
        else:  # braille_e
            # Average learner - typical pattern
            reviews = [(0, 4), (1, 4), (7, 3), (15, 4), (30, 4)]
        
        for day, rating in reviews:
            decision = scheduler.schedule_review(
                user_id=user_id,
                item_id=char_id,
                rating=rating,
                timestamp=day
            )
            
            state = scheduler.get_state(user_id, char_id)
            params = state.algorithm_params
            
            print(f"  Day {day:2d} | Rating={rating} | Next: {decision.interval:.0f} days | "
                  f"EF={params['ease_factor']:.2f}")
        
        # Calculate final half-life
        half_life = scheduler.calculate_half_life(user_id, char_id)
        print(f"  Final half-life: {half_life:.1f} days\n")
    
    # Show overall statistics
    print("--- Overall Learning Statistics ---")
    stats = scheduler.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")


def dataset_integration_example():
    """Show how to integrate with FSRS-Anki-20k dataset."""
    print("\n=== Dataset Integration Example ===\n")
    
    print("When the FSRS-Anki-20k dataset is loaded:")
    print("""
# Example integration code:

from datasets import load_dataset
from src.schedulers import SM2Scheduler, RatingConverter

# Load dataset
ds = load_dataset("open-spaced-repetition/FSRS-Anki-20k")

# Initialize scheduler
scheduler = SM2Scheduler()

# Process user reviews
for user_data in ds['train']:
    user_id = user_data['user_id']
    
    for review in user_data['reviews']:
        item_id = review['card_id']
        anki_rating = review['rating']  # 1-4 scale
        timestamp = review['timestamp']
        
        # Convert Anki rating to SM-2 quality
        sm2_quality = RatingConverter.anki_to_sm2(anki_rating)
        
        # Process with SM-2
        decision = scheduler.schedule_review(
            user_id=user_id,
            item_id=item_id,
            rating=sm2_quality,
            timestamp=timestamp
        )
        
        # Store results for evaluation
        # ... (save decision.interval, decision.p_recall, etc.)
""")
    
    print("This will allow the system to:")
    print("1. Process thousands of real learning sessions")
    print("2. Compare SM-2 performance with other algorithms")
    print("3. Calculate evaluation metrics (SRP, THR, WTL, etc.)")
    print("4. Optimize for Braille learning specifically")


def rating_conversion_demo():
    """Show rating conversion between different systems."""
    print("\n=== Rating Conversion Demo ===\n")
    
    print("Converting between Anki (1-4) and SM-2 (0-5) scales:")
    print()
    
    print("Anki → SM-2:")
    for anki in [1, 2, 3, 4]:
        sm2 = RatingConverter.anki_to_sm2(anki)
        meanings = ["Again (failed)", "Hard", "Good", "Easy"]
        print(f"  Anki {anki} ({meanings[anki-1]}) → SM-2 {sm2}")
    
    print("\nSM-2 → Anki:")
    for sm2 in [0, 1, 2, 3, 4, 5]:
        anki = RatingConverter.sm2_to_anki(sm2)
        meanings = ["Complete blackout", "Incorrect", "Incorrect", "Difficult", "Good", "Perfect"]
        print(f"  SM-2 {sm2} ({meanings[sm2]}) → Anki {anki}")


if __name__ == "__main__":
    print("SM-2 Scheduler Demo for Braille Learning")
    print("=" * 45)
    
    # Run all demos
    simple_usage_demo()
    braille_learning_scenario()
    dataset_integration_example()
    rating_conversion_demo()
    
    print("\n" + "="*45)
    print("🎉 Demo completed!")
    print("\nThe SM-2 scheduler is ready for Braille learning optimization.")
    print("Next: Download the FSRS-Anki-20k dataset and run evaluation!")
