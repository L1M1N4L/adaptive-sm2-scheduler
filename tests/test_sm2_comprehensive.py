"""
SM-2 Scheduler Test Suite

This file tests the SM-2 implementation to make sure it works correctly.
Run this file to verify everything is working properly.
"""

import sys
import os
import math
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.schedulers import SM2Scheduler, SM2StandaloneCard, RatingConverter


def test_sm2_algorithm():
    """Test that the SM-2 algorithm works correctly."""
    print("Testing SM-2 Algorithm...")
    
    # Create a new card
    card = SM2StandaloneCard()
    
    # Test first review
    interval = card.review(5)  # Perfect score
    assert interval == 1, f"First review should give 1 day, got {interval}"
    
    # Test second review
    interval = card.review(4)  # Good score
    assert interval == 6, f"Second review should give 6 days, got {interval}"
    
    # Test third review
    interval = card.review(5)  # Perfect score
    expected = math.ceil(6 * 2.6)  # 6 * 2.6 = 15.6, rounded up = 16
    assert interval == expected, f"Third review should give {expected} days, got {interval}"
    
    print("✓ SM-2 algorithm test passed")


def test_rating_conversion():
    """Test that rating conversion works correctly."""
    print("Testing Rating Conversion...")
    
    # Test Anki to SM-2 conversion
    assert RatingConverter.anki_to_sm2(1) == 0, "Anki 1 should convert to SM-2 0"
    assert RatingConverter.anki_to_sm2(2) == 3, "Anki 2 should convert to SM-2 3"
    assert RatingConverter.anki_to_sm2(3) == 4, "Anki 3 should convert to SM-2 4"
    assert RatingConverter.anki_to_sm2(4) == 5, "Anki 4 should convert to SM-2 5"
    
    # Test SM-2 to Anki conversion
    assert RatingConverter.sm2_to_anki(0) == 1, "SM-2 0 should convert to Anki 1"
    assert RatingConverter.sm2_to_anki(3) == 2, "SM-2 3 should convert to Anki 2"
    assert RatingConverter.sm2_to_anki(4) == 3, "SM-2 4 should convert to Anki 3"
    assert RatingConverter.sm2_to_anki(5) == 4, "SM-2 5 should convert to Anki 4"
    
    print("✓ Rating conversion test passed")


def test_scheduler_interface():
    """Test the main scheduler interface."""
    print("Testing Scheduler Interface...")
    
    scheduler = SM2Scheduler()
    
    # Test first review
    decision = scheduler.schedule_review("user1", "item1", 4, 0.0)
    assert decision.interval == 1, f"First review should give 1 day, got {decision.interval}"
    assert decision.repetitions == 1, f"First review should have 1 repetition, got {decision.repetitions}"
    
    # Test second review
    decision = scheduler.schedule_review("user1", "item1", 5, 1.0)
    assert decision.interval == 6, f"Second review should give 6 days, got {decision.interval}"
    assert decision.repetitions == 2, f"Second review should have 2 repetitions, got {decision.repetitions}"
    
    # Test recall prediction
    p_recall = scheduler.predict_recall("user1", "item1", 3.0)
    assert 0 <= p_recall <= 1, f"Recall probability should be between 0 and 1, got {p_recall}"
    
    # Test half-life calculation
    half_life = scheduler.calculate_half_life("user1", "item1")
    assert half_life > 0, f"Half-life should be positive, got {half_life}"
    
    print("✓ Scheduler interface test passed")


def test_braille_learning_scenario():
    """Test a realistic Braille learning scenario."""
    print("Testing Braille Learning Scenario...")
    
    scheduler = SM2Scheduler()
    
    # Simulate learning Braille characters
    braille_chars = ['braille_a', 'braille_b', 'braille_c']
    
    for char in braille_chars:
        # First review - learning the character
        decision = scheduler.schedule_review("learner_001", char, 4, 0.0)
        assert decision.interval == 1, f"First review of {char} should give 1 day"
        
        # Second review - after 1 day
        decision = scheduler.schedule_review("learner_001", char, 5, 1.0)
        assert decision.interval == 6, f"Second review of {char} should give 6 days"
        
        # Third review - after 6 days
        decision = scheduler.schedule_review("learner_001", char, 4, 7.0)
        assert decision.interval > 6, f"Third review of {char} should give more than 6 days"
    
    print("✓ Braille learning scenario test passed")


def test_performance():
    """Test that the scheduler performs well with many items."""
    print("Testing Performance...")
    
    scheduler = SM2Scheduler()
    start_time = time.time()
    
    # Test with 1000 items
    for i in range(1000):
        user_id = f"user_{i % 10}"  # 10 different users
        item_id = f"item_{i}"
        
        # Simulate 5 reviews per item
        for review_num in range(5):
            rating = 4 if review_num < 3 else 5  # Mix of good and perfect ratings
            timestamp = review_num * 1.0
            scheduler.schedule_review(user_id, item_id, rating, timestamp)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Processed 1000 items with 5 reviews each in {elapsed:.3f} seconds")
    print(f"Average time per review: {elapsed / 5000 * 1000:.2f} ms")
    
    # Test statistics
    stats = scheduler.get_statistics()
    assert stats['total_reviews'] == 5000, f"Expected 5000 reviews, got {stats['total_reviews']}"
    assert stats['total_items'] == 1000, f"Expected 1000 items, got {stats['total_items']}"
    
    print("✓ Performance test passed")


def run_all_tests():
    """Run all tests and show results."""
    print("SM-2 Scheduler Test Suite")
    print("=" * 40)
    print()
    
    try:
        test_sm2_algorithm()
        print()
        
        test_rating_conversion()
        print()
        
        test_scheduler_interface()
        print()
        
        test_braille_learning_scenario()
        print()
        
        test_performance()
        print()
        
        print("🎉 All tests passed!")
        print("The SM-2 scheduler is working correctly and ready for Braille learning.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)