# Phase 1: CUDA Context Fixes - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning
**Mode:** Auto-generated (discuss skipped via workflow.skip_discuss)

<domain>
## Phase Boundary

Add cudaSetDevice(0) to all test fixtures that need it to prevent SEGFAULT errors. Target fixtures:
- JacobiPreconditionerTest (11 tests)
- RCMReordererTest (13 tests)
- PreconditionedSolverTest (6 tests)
- SSSPTest (4 tests)
- MemoryNodeTest (8 tests)
- GraphExecutorTest (3 tests)

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
All implementation choices are at the agent's discretion — pure infrastructure phase. Use ROADMAP phase goal and success criteria to guide decisions.

</decisions>

<codebase>
## Existing Code Insights

### Pattern Observed
Test fixtures currently skip with `GTEST_SKIP() << "JacobiPreconditioner tests have CUDA context issues - skipping"` in SetUp(). Replace skip with proper CUDA device initialization.

### Files to Modify
- tests/sparse/preconditioner_test.cpp
- tests/sparse/reordering_test.cpp
- tests/sparse/preconditioned_solver_test.cpp
- tests/algo/sssp_test.cpp
- tests/cuda/production/memory_node_test.cpp
- tests/cuda/production/graph_executor_test.cpp

</codebase>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to ROADMAP phase description and success criteria:
- Add cudaSetDevice(0) to SetUp() methods
- Remove GTEST_SKIP() calls
- Add cudaDeviceSynchronize() if not already present
- Verify 45 tests pass without SEGFAULT

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase with clear scope.

</deferred>
