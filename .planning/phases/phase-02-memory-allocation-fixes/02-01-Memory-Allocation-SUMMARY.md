---
phase: "02"
plan_id: "02-01"
plan_name: "Memory Allocation Fixes"
status: "complete"
completed_at: "2026-05-09"
wave: 1
tasks_completed: 1
tasks_total: 1
---

# Plan 02-01: Memory Allocation Fixes

## Summary

Reduced memory allocations in block_manager_edge_test.cpp to prevent OOM errors during testing.

## Tasks Completed

1. **Memory Allocation Reductions** - tests/inference/block_manager_edge_test.cpp
   - Reduced `num_gpu_blocks` from 8192 to 256
   - Reduced `max_model_len` from 32768 to 8192
   - Un-skipped LongPromptKVCacheStats test

## Files Modified

- tests/inference/block_manager_edge_test.cpp

## Tests Fixed

7 tests updated:

- ChunkedPrefillTest::LongPromptStressTest
- ChunkedPrefillTest::LongSequenceWithBlockGrowth
- LongPromptIntegrationTest::PromptOver16KTokens
- LongPromptIntegrationTest::MultipleLongSequences
- LongPromptIntegrationTest::LongPromptForwardBatch
- LongPromptIntegrationTest::LongPromptKVCacheStats (was GTEST_SKIP, now enabled)
- LongPromptIntegrationTest::LongPromptSequenceIsolation
