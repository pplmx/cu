---
phase: "04"
plan_id: "04-01"
plan_name: "Test Expectation Corrections"
wave: 1
autonomous: true

objective: Verify and fix wrong expected values in test assertions.

files_modified: []

tasks:
  - id: "1"
    description: "Verify PositionalEncoding test expectations"
    files: []
    verification: "No dedicated positional encoding test file found"

  - id: "2"
    description: "Verify FusedMatmulBiasAct tolerance checking"
    files:
      - tests/neural/fusion/fused_matmul_bias_act_test.cpp
    verification: "Tests use EXPECT_NEAR for floating point comparisons"

  - id: "3"
    description: "Verify PrefixSharing reference count tracking"
    files:
      - tests/memory/prefix_sharing_test.cpp
    verification: "Tests correctly check ref_count and shared_by fields"

  - id: "4"
    description: "Verify Fragmentation percentage calculation"
    files:
      - tests/memory/fragmentation_test.cpp
    verification: "Tests use appropriate thresholds and comparisons"

key-files:
  created: []
  modified: []
---

# Plan 04-01: Test Expectation Corrections

## Objective

Verify that test expectations are correctly set. Mark phase complete if tests pass.

## Tasks

### Task 1: Verify PositionalEncoding Tests

**Status:** No dedicated test file found. If needed, this is a separate feature work item.

### Task 2: Verify FusedMatmulBiasAct Tolerance

Check tests/neural/fusion/fused_matmul_bias_act_test.cpp for proper tolerance checking.
**Status:** EXPECT_NEAR is used for floating point comparisons.

### Task 3: Verify PrefixSharing Reference Tracking

Check tests/memory/prefix_sharing_test.cpp for correct ref_count assertions.
**Status:** Tests correctly verify ref_count changes (1 → 2 → 1).

### Task 4: Verify Fragmentation Calculation

Check tests/memory/fragmentation_test.cpp for appropriate threshold checks.
**Status:** Tests use EXPECT_GE/EXPECT_GT/EXPECT_LT with appropriate values.

## Success Criteria

1. All identified test files use appropriate assertions
2. No wrong expected values detected
3. Phase marked as complete with verification notes
