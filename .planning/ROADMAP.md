# Roadmap: v2.15 Test Quality Assurance

**Created:** 2026-05-09
**Phases:** 5
**Requirements mapped:** 21

## Overview

| Phase | Name | Goal | Requirements | Success Criteria |
|-------|------|------|--------------|------------------|
| 1 | CUDA Context Fixes | Add cudaSetDevice to all test fixtures | CTX-01, CTX-02, CTX-03 | 40+ test fixtures updated, tests pass |
| 2 | Memory Allocation Fixes | Reduce test memory usage | MEM-01, MEM-02, MEM-03, MEM-04 | OOM tests pass with reduced allocation |
| 3 | Algorithm Kernel Fixes | Fix broken kernel implementations | ALGO-01, ALGO-02, ALGO-03, ALGO-04 | FlashAttention, TopK, SegmentedSort pass |
| 4 | Test Expectation Corrections | Fix wrong expected values | TEST-01, TEST-02, TEST-03, TEST-04 | Correct values in tests |
| 5 | Error Handling & Safety | Fix error/memory test issues | ERR-01, ERR-02, ERR-03, ERR-04, SAFE-01, SAFE-02, SAFE-03 | All error handling tests pass |

---

## Phase 1: CUDA Context Fixes

**Goal:** Add cudaSetDevice(0) to all test fixtures that need it

**Requirements:** CTX-01, CTX-02, CTX-03

**Success Criteria:**
1. JacobiPreconditionerTest (11 tests) - add SetUp with cudaSetDevice
2. RCMReordererTest (13 tests) - add SetUp with cudaSetDevice  
3. PreconditionedSolverTest (6 tests) - add SetUp with cudaSetDevice
4. SSSPTest (4 tests) - add SetUp with cudaSetDevice
5. MemoryNodeTest (8 tests) - add SetUp with cudaSetDevice
6. GraphExecutorTest (3 tests) - add SetUp with cudaSetDevice
7. Verify tests no longer SEGFAULT
8. All 45 tests pass

**Files to modify:**
- `tests/sparse/preconditioner_test.cpp`
- `tests/sparse/reordering_test.cpp`
- `tests/sparse/preconditioned_solver_test.cpp`
- `tests/algo/sssp_test.cpp`
- `tests/cuda/production/memory_node_test.cpp`
- `tests/cuda/production/graph_executor_test.cpp`

---

## Phase 2: Memory Allocation Fixes

**Goal:** Reduce test memory usage to fit within GPU memory limits

**Requirements:** MEM-01, MEM-02, MEM-03, MEM-04

**Success Criteria:**
1. BlockManager tests - reduce num_gpu_blocks from 8192 to 256
2. DynamicBlockSizing tests - reduce max_model_len from 32768 to 8192
3. BeamSearch tests - reduce beam width or block allocation
4. ChunkedPrefill tests - use smaller prefill chunks
5. All memory-intensive tests pass without OOM
6. Verify functionality still works with smaller allocations

**Files to modify:**
- `tests/inference/block_manager_edge_test.cpp`
- `tests/inference/beam_search_test.cpp`
- `tests/inference/chunked_prefill_test.cpp`

---

## Phase 3: Algorithm Kernel Fixes

**Goal:** Fix broken kernel implementations in core algorithms

**Requirements:** ALGO-01, ALGO-02, ALGO-03, ALGO-04

**Success Criteria:**
1. FlashAttention - fix causal masking in kernel
   - Verify causal output differs from non-causal
   - Test with various sequence lengths
2. TopK - fix selection algorithm
   - Use CUB top-k or proper sorting
   - Verify returns actual top K elements
3. SegmentedSort - fix segment boundary handling
   - Verify segments maintain correct boundaries
4. StreamingCache - fix eviction logic
   - Verify LRU eviction works correctly

**Files to modify:**
- `src/cuda/algo/flash_attention.cu`
- `src/cuda/algo/sort.cu` (select_top_k)
- `src/cuda/algo/segmented_sort.cu`
- `src/cuda/inference/streaming_cache.cpp`

---

## Phase 4: Test Expectation Corrections

**Goal:** Fix wrong expected values in tests

**Requirements:** TEST-01, TEST-02, TEST-03, TEST-04

**Success Criteria:**
1. PositionalEncoding - verify correct encoding formula
   - Test with known inputs
   - Compare against numpy implementation
2. FusedMatmulBiasAct - verify correct output shapes
   - Add tolerance checking for floating point
3. PrefixSharing - verify reference count tracking
   - Add debug output to trace fork/merge
4. Fragmentation - verify percentage calculation
   - Test with known allocation patterns

**Files to modify:**
- `tests/algo/positional_encoding_test.cpp`
- `tests/algo/fused_matmul_bias_act_test.cpp`
- `tests/inference/prefix_sharing_test.cpp`
- `tests/inference/fragmentation_test.cpp`

---

## Phase 5: Error Handling & Safety

**Goal:** Fix error handling and memory safety test issues

**Requirements:** ERR-01, ERR-02, ERR-03, ERR-04, SAFE-01, SAFE-02, SAFE-03

**Success Criteria:**
1. TimeoutPropagation - verify timeout callbacks work
   - Test with short timeouts
   - Verify callback invoked
2. RetryTest - fix circuit breaker state machine
   - Verify closed→half-open→open transitions
3. HierarchicalAllReduce - handle null communicators
   - Add early return for null comms
4. ErrorInjection - verify error detection works
   - Test with known error patterns
5. MemorySafetyTest - verify poison detection
   - Test with uninitialized memory patterns
6. AttentionSink - verify sink block tracking
   - Test sink promotion/demotion
7. MemoryNodeTest - verify allocation types
   - Test device/host/managed allocations

**Files to modify:**
- `tests/cuda/error/timeout_propagation_test.cpp`
- `tests/cuda/error/retry_test.cpp`
- `tests/cuda/distributed/hierarchical_all_reduce_test.cpp`
- `tests/cuda/testing/error_injection_test.cpp`
- `tests/cuda/testing/memory_safety_test.cpp`
- `tests/inference/attention_sink_test.cpp`
- `tests/cuda/production/memory_node_test.cpp`

---

## Summary

| Phase | Requirements | Focus |
|-------|--------------|-------|
| 1 | CTX-01, CTX-02, CTX-03 | Test infrastructure |
| 2 | MEM-01, MEM-02, MEM-03, MEM-04 | Memory optimization |
| 3 | ALGO-01, ALGO-02, ALGO-03, ALGO-04 | Kernel fixes |
| 4 | TEST-01, TEST-02, TEST-03, TEST-04 | Test correctness |
| 5 | ERR-01, ERR-02, ERR-03, ERR-04, SAFE-01, SAFE-02, SAFE-03 | Error handling |

**Total:** 5 phases, 21 requirements

**Target:** 100% pass rate (from ~85%), 0 skips for fixable tests
