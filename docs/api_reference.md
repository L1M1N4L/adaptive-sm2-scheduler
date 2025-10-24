# API Reference

This document provides comprehensive API documentation for the Adaptive SM-2 Scheduler project. The implementation follows a professional, scalable architecture designed for research applications.

## Project Structure

The project is organized into logical directories:

- **`src/schedulers/`**: Core algorithm implementations
- **`scripts/`**: Evaluation and analysis scripts
- **`evaluation/`**: SSP-MMC-Plus framework and results
- **`tests/`**: Comprehensive test suite
- **`examples/`**: Usage examples and demos
- **`docs/`**: Complete documentation

## BaseScheduler

Abstract base class for all spaced repetition schedulers.

### Constructor

```python
BaseScheduler(name: str)
```

Initialize scheduler with a name identifier.

### Abstract Methods

#### schedule_review

```python
schedule_review(user_id: str, item_id: str, rating: int, timestamp: float, history: Optional[List[Tuple[float, int]]] = None) -> ScheduleDecision
```

Process a review and determine next review schedule.

**Parameters:**
- `user_id`: Unique user identifier
- `item_id`: Unique item/card identifier  
- `rating`: Review outcome/rating (0-5 for SM-2)
- `timestamp`: Current review timestamp (days)
- `history`: Optional list of (timestamp, rating) tuples

**Returns:** ScheduleDecision with interval, p_recall, and confidence

#### predict_recall

```python
predict_recall(user_id: str, item_id: str, delta_t: float) -> float
```

Predict recall probability after delta_t days.

**Parameters:**
- `user_id`: User identifier
- `item_id`: Item identifier
- `delta_t`: Days since last review

**Returns:** Predicted recall probability [0, 1]

#### calculate_half_life

```python
calculate_half_life(user_id: str, item_id: str) -> float
```

Calculate memory half-life for an item.

**Parameters:**
- `user_id`: User identifier
- `item_id`: Item identifier

**Returns:** Half-life in days

### Concrete Methods

#### get_state

```python
get_state(user_id: str, item_id: str) -> Optional[SchedulerState]
```

Get current state for a user-item pair.

#### set_state

```python
set_state(user_id: str, item_id: str, state: SchedulerState) -> None
```

Set state for a user-item pair.

#### reset

```python
reset() -> None
```

Reset all scheduler states.

#### get_statistics

```python
get_statistics() -> Dict[str, float]
```

Get statistics about current scheduler state.

**Returns:** Dictionary with total_items, avg_interval, median_interval, etc.

#### get_all_half_lives

```python
get_all_half_lives() -> Dict[Tuple[str, str], float]
```

Get half-lives for all tracked items.

**Returns:** Dictionary mapping (user_id, item_id) to half_life

#### get_all_recall_predictions

```python
get_all_recall_predictions(delta_t: float) -> Dict[Tuple[str, str], float]
```

Get recall predictions for all tracked items at delta_t days.

**Parameters:**
- `delta_t`: Days since last review

**Returns:** Dictionary mapping (user_id, item_id) to recall_probability

## SM2Scheduler

Implementation of the SM-2 spaced repetition algorithm.

### Constructor

```python
SM2Scheduler()
```

Initialize SM-2 scheduler with default parameters.

### Constants

- `DEFAULT_EF = 2.5`: Default ease factor
- `MIN_EF = 1.3`: Minimum ease factor
- `FIRST_INTERVAL = 1`: First review interval (days)
- `SECOND_INTERVAL = 6`: Second review interval (days)

### Methods

Inherits all methods from BaseScheduler. Additional methods:

#### _sm2_algorithm

```python
_sm2_algorithm(quality: int, repetitions: int, ease_factor: float, interval: float) -> Tuple[float, int, float]
```

Core SM-2 algorithm logic.

**Parameters:**
- `quality`: Quality score 0-5
- `repetitions`: Number of successful reviews
- `ease_factor`: Current ease factor
- `interval`: Previous interval

**Returns:** Tuple of (new_interval, new_repetitions, new_ease_factor)

## SM2StandaloneCard

Standalone SM-2 card for simple use cases (backwards compatible).

### Constructor

```python
SM2StandaloneCard(ease_factor: float = None, interval: int = 0, repetitions: int = 0)
```

Initialize card with optional previous state.

### Methods

#### review

```python
review(quality: int) -> int
```

Process review and return next interval.

**Parameters:**
- `quality`: Quality score 0-5

**Returns:** Next interval in days

#### get_state

```python
get_state() -> Dict[str, float]
```

Get current state.

**Returns:** Dictionary with ease_factor, interval, repetitions

## RatingConverter

Utility class for converting between rating systems.

### Static Methods

#### anki_to_sm2

```python
anki_to_sm2(anki_rating: int) -> int
```

Convert Anki rating (1-4) to SM-2 quality (0-5).

**Parameters:**
- `anki_rating`: Anki rating (1-4)

**Returns:** SM-2 quality (0-5)

#### sm2_to_anki

```python
sm2_to_anki(sm2_quality: int) -> int
```

Convert SM-2 quality (0-5) to Anki rating (1-4).

**Parameters:**
- `sm2_quality`: SM-2 quality (0-5)

**Returns:** Anki rating (1-4)

#### normalize_rating

```python
normalize_rating(rating: int, source_scale: str = "anki", target_scale: str = "sm2") -> int
```

Normalize rating between different systems.

**Parameters:**
- `rating`: Input rating
- `source_scale`: "anki" or "sm2"
- `target_scale`: "anki" or "sm2"

**Returns:** Converted rating

## Data Classes

### ReviewRecord

Single review record in SSP-MMC-Plus format.

**Attributes:**
- `u`: user_id
- `w`: word/item_id
- `i`: review_index (0-based)
- `d`: initial difficulty/rating
- `t_history`: list of review timestamps
- `r_history`: list of review ratings/outcomes
- `delta_t`: time since last review (days)
- `r`: current rating/outcome
- `p_recall`: predicted recall probability [0, 1]
- `total_cnt`: total number of reviews so far

### SchedulerState

Internal state of a scheduler for a specific item.

**Attributes:**
- `item_id`: Item identifier
- `user_id`: User identifier
- `last_review`: Last review timestamp
- `next_review`: Next review timestamp
- `algorithm_params`: Algorithm-specific parameters

### ScheduleDecision

Output of a scheduling decision.

**Attributes:**
- `interval`: Days until next review
- `p_recall`: Predicted recall probability at next review time
- `confidence`: Scheduler's confidence in this prediction [0, 1]
- `metadata`: Additional algorithm-specific information

## Utility Functions

### exponential_forgetting_curve

```python
exponential_forgetting_curve(t: float, S: float) -> float
```

Calculate recall probability using exponential forgetting curve.

**Formula:** R(t) = 2^(-t/S)

**Parameters:**
- `t`: Days since last review
- `S`: Memory stability (half-life in days)

**Returns:** Recall probability [0, 1]

### calculate_half_life_from_recall

```python
calculate_half_life_from_recall(delta_t: float, p_recall: float) -> float
```

Calculate half-life from observed recall probability.

**Parameters:**
- `delta_t`: Time since review (days)
- `p_recall`: Observed recall probability

**Returns:** Estimated half-life (days)
