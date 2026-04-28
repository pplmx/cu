# Phase 64: Timeout Foundation - Plan

## Requirements
- TO-01: Per-operation timeout tracking with configurable deadlines
- TO-02: Watchdog timer system for detecting stalled operations

## Implementation Plan

### 1. Create TimeoutContext class
- RAII-managed timeout tracking per operation
- Stores deadline, operation name, start time
- Integrates with std::chrono for timeout durations

### 2. Create TimeoutManager class
- Singleton managing all active operations
- Thread-safe operation tracking (std::unordered_map + mutex)
- Watchdog background thread for detecting timeouts

### 3. Create timeout_error_category
- Extends std::error_category for timeout errors
- Uses errc::timed_out as base
- Provides descriptive timeout messages

### 4. Add tests
- Timeout detection unit tests
- Watchdog thread tests
- Concurrent operation timeout tests

## Files to Create/Modify
- `include/cuda/error/timeout.hpp` (new)
- `src/error/timeout.cpp` (new)
- `tests/error/timeout_test.cpp` (new)
- `src/CMakeLists.txt` (add timeout.cpp)
- `tests/CMakeLists.txt` (add test)
