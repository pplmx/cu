# Phase 2: Memory Allocation Fixes - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning
**Mode:** Auto-generated (discuss skipped via workflow.skip_discuss)

<domain>
## Phase Boundary

Reduce test memory usage to fit within GPU memory limits. Target high allocation tests:
- BlockManager tests with 8192 blocks → reduce to 256
- DynamicBlockSizing tests with high max_model_len → reduce to reasonable values
- BeamSearch tests with excessive allocations
- ChunkedPrefill tests handling memory limits

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
All implementation choices are at the agent's discretion — pure infrastructure phase. Reduce allocations to fit standard GPU memory while keeping tests functional.

</decisions>

<codebase>
## Existing Code Insights

### High Allocation Tests Found
block_manager_edge_test.cpp has tests with:
- Line 560-561: num_gpu_blocks = 8192, max_model_len = 16384
- Line 597-598: num_gpu_blocks = 8192, max_model_len = 32768
- Line 646-647: num_gpu_blocks = 8192, max_model_len = 32768
- Line 672-673: num_gpu_blocks = 8192, max_model_len = 32768

</codebase>

<specifics>
## Specific Ideas

Per ROADMAP Phase 2:
- Reduce num_gpu_blocks from 8192 to 256
- Reduce max_model_len from 32768 to 8192
- Ensure tests still pass with reduced allocations

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase with clear scope.

</deferred>