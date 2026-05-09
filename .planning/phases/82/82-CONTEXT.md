# Phase 82: Integration & Production - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>

## Phase Boundary

Users can reuse solver workspace, access diagnostics, run comprehensive tests and benchmarks, and access documentation. This phase integrates all prior work into a production-ready system.

</domain>

<decisions>

## Implementation Decisions

### Workspace Memory Pool (KRY-05)

- Create `SolverWorkspace<T>` class that wraps pre-allocated work vectors
- Reuses allocation across multiple solve() calls
- Integrates with existing nova memory pool if available
- Fallback to standard allocation if pool unavailable

### Solver Diagnostics (KRY-06)

- SolverResult already includes: iterations, residual_norm, residual_history
- Add diagnostic methods to extract timing, convergence rate
- Provide access to residual history for external analysis

### E2E Tests (INT-01)

- Integration test covering: CSR → ELL → SpMV → CG solver → Roofline
- Test full pipeline from matrix creation to solution verification
- Validate all format conversions produce correct results

### Benchmarks (INT-02)

- Benchmark comparing SpMV performance: CSR, ELL, SELL, HYB
- Benchmark solver convergence for varying matrix sizes
- Record and report performance metrics

### NVTX Annotations (INT-03)

- Add NVTX domain "nova_sparse" for all sparse operations
- Annotate SpMV, solver iteration loops, format conversions
- Enable with compile-time flag NOVA_NVTX

### Documentation (INT-04)

- Update sparse formats documentation (sparse_formats.md)
- Add Krylov solver documentation (krylov_solvers.md)
- Add Roofline model documentation (roofline.md)
- Update index with new components

### the agent's Discretion

- Memory pool implementation details
- Test matrix generation strategies
- Benchmark reporting format

</decisions>

<codebase>

## Existing Patterns

From Phase 75-78 (NVTX, profiling):

- NVTX domain creation: `nvtxDomainHandle_t`
- Range annotation: `nvtxRangePush/Pop`
- Compile-time enable: `#ifdef NOVA_NVTX`

From memory pool (prior milestones):

- Buffer reuse patterns
- Allocation tracking

</codebase>

<specifics>

## Specific Ideas

No specific requirements — follow standard approaches:

- Workspace pooling: common optimization pattern
- NVTX: standard NVIDIA tooling
- Documentation: clear examples with expected output

</specifics>

<deferred>

## Deferred Ideas

None — all requirements addressed in this phase

</deferred>
