# Phase 3: Algorithm Kernel Fixes - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning
**Mode:** Auto-generated (discuss skipped via workflow.skip_discuss)

<domain>
## Phase Boundary

Fix broken kernel implementations in core algorithms:
- FlashAttention causal masking verification
- TopK selection correctness
- SegmentedSort kernel fixes
- StreamingCache eviction logic

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
Implementation choices at agent's discretion. Add cudaSetDevice to test fixtures and verify kernel behavior.

</decisions>

<codebase>
## Existing Code Insights

### SegmentedSort Test Status
tests/algo/segmented_sort_test.cpp has GTEST_SKIP() due to CUDA context issues - needs cudaSetDevice(0)

### Files to Examine
- src/algo/flash_attention.cu
- src/cuda/algo/sort.cu
- src/algo/segmented_sort.cu
- src/cuda/memory/streaming_cache_manager.cpp

</codebase>

<specifics>
## Specific Ideas

Per ROADMAP Phase 3:
- Add cudaSetDevice to SegmentedSortTest fixture
- Verify kernel behavior for all algorithm tests

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase with clear scope.

</deferred>