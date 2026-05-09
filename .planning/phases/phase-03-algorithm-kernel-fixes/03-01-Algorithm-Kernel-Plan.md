---
phase: "03"
plan_id: "03-01"
plan_name: "Algorithm Kernel Fixes"
wave: 1
autonomous: true

objective: Enable SegmentedSortTest by adding cudaSetDevice(0) to the fixture.

files_modified:
  - tests/algo/segmented_sort_test.cpp

tasks:
  - id: "1"
    description: "Fix SegmentedSortTest fixture"
    files:
      - tests/algo/segmented_sort_test.cpp
    verification: "GTEST_SKIP removed, cudaSetDevice(0) added, tests pass"

key-files:
  created: []
  modified:
    - tests/algo/segmented_sort_test.cpp
---

# Plan 03-01: Algorithm Kernel Fixes

## Objective

Enable SegmentedSortTest by adding proper CUDA context initialization.

## Tasks

### Task 1: Fix SegmentedSortTest fixture

**File:** tests/algo/segmented_sort_test.cpp

**Change:** Replace `GTEST_SKIP()` with proper CUDA device initialization:

```cpp
void SetUp() override {
    cudaSetDevice(0);
    cudaDeviceSynchronize();
}
```

## Success Criteria

1. SegmentedSortTest no longer skips
2. All 5 tests in the fixture run to completion
