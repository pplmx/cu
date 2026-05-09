---
phase: "02"
plan_id: "02-01"
plan_name: "Memory Allocation Fixes"
wave: 1
autonomous: true

objective: Reduce test memory allocations to fit within GPU memory limits. Target high allocations in BlockManager tests (8192 blocks → 256 blocks).

files_modified:
  - tests/inference/block_manager_edge_test.cpp

tasks:
  - id: "1"
    description: "Reduce high allocation tests in block_manager_edge_test.cpp"
    files:
      - tests/inference/block_manager_edge_test.cpp
    verification: "All tests pass with reduced memory allocations"

key-files:
  created: []
  modified:
    - tests/inference/block_manager_edge_test.cpp
---

# Plan 02-01: Memory Allocation Fixes

## Objective

Reduce memory allocations in block_manager_edge_test.cpp to prevent OOM errors during testing.

## Target Reductions

| Test Config | Current | Target |
|-------------|---------|--------|
| num_gpu_blocks | 8192 | 256 |
| max_model_len | 32768 | 8192 |

## Tasks

### Task 1: Fix High Allocation Tests in block_manager_edge_test.cpp

Review all test configurations in tests/inference/block_manager_edge_test.cpp and reduce allocations to reasonable values:

- Tests with `num_gpu_blocks = 8192` → reduce to 256
- Tests with `max_model_len = 32768` → reduce to 8192
- Tests with `max_model_len = 16384` → reduce to 8192

Key locations (approximately):

- Lines 560-576: OOM stress test
- Lines 597-618: Multiple sequence test
- Lines 646-678: Large model test

## Success Criteria

1. All BlockManager tests pass without OOM
2. No test config has num_gpu_blocks > 256
3. No test config has max_model_len > 8192
