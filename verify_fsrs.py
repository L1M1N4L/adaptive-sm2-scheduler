
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

try:
    from src.schedulers.fsrs import FSRSScheduler
    print("SUCCESS: FSRSScheduler imported successfully.")
except ImportError as e:
    print(f"FAILURE: Could not import FSRSScheduler. Error: {e}")
    sys.exit(1)

scheduler = FSRSScheduler()
user_id = "test_user"
item_id = "test_item"
timestamp = 1.0

print(f"Testing schedule_review for new item...")
decision = scheduler.schedule_review(user_id, item_id, 4, timestamp)
print(f"Decision: Interval={decision.interval}, Ease={decision.ease_factor:.2f}")

if decision.interval > 0:
    print("SUCCESS: Valid interval calculated.")
else:
    print("FAILURE: Interval calculated is 0 or less.")

# Simulate a second review
timestamp += decision.interval
print(f"Testing second review (Space: {decision.interval} days)...")
decision2 = scheduler.schedule_review(user_id, item_id, 3, timestamp)
print(f"Decision 2: Interval={decision2.interval}, Ease={decision2.ease_factor:.2f}")

if decision2.interval >= decision.interval:
    print("SUCCESS: Interval increased or stayed same (as expected for 'Hard' on FSRS).")
else:
    print("WARNING: Interval decreased. (Might be valid depending on logic, but checking).")

print("Verification complete.")
