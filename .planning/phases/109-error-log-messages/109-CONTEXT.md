# Phase 109: Error/Log Messages - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>

## Phase Boundary

Improve structured logging and error diagnostics across the codebase. Ensure all error messages include actionable guidance, implement proper log levels, and add context to diagnostics.
</domain>

<decisions>

## Implementation Decisions

### Log Level Strategy

- **L-01:** Use ERROR for failures that prevent operation
- **L-02:** Use WARN for recoverable issues (degraded performance, fallback)
- **L-03:** Use INFO for significant milestones (allocation complete, training epoch)
- **L-04:** Use DEBUG for detailed trace information
- **L-05:** Use TRACE for per-iteration logging (disabled by default)

### Error Message Format

- **L-06:** Error messages include: what failed, why it failed, how to fix
- **L-07:** Include operation name, device/stream context, relevant values
- **L-08:** Link to documentation where applicable

### Structured Logging

- **L-09:** Use key=value pairs for structured context
- **L-10:** Consistent field names across all log messages
- **L-11:** Include timestamps in ISO 8601 format

</decisions>

<canonical_refs>

## Canonical References

### Logging Infrastructure

- `.planning/codebase/CONVENTIONS.md` §Logging — Logging style guide
- `include/cuda/observability/nvtx.hpp` — NVTX profiling markers

### Error Handling (Phase 107)

- `include/cuda/error/cuda_error.hpp` — Error structure
- `include/cuda/error/retry.hpp` — Retry mechanisms

### Related Phases

- Phase 107 (API Documentation) — Error type documentation
- Phase 108 (Code Comments) — Inline comments
</canonical_refs>

<codebase_context>

## Existing Code Insights

### Current Error Handling

- Error codes defined in include/cuda/error/*.hpp
- Recovery hints implemented in cuda_error.hpp
- Retry logic in retry.hpp

### Current Logging

- NOVA_CHECK and CUBLAS_CHECK macros for error propagation
- Limited structured logging in existing code
- No consistent log level usage

### Areas Needing Improvement

1. Sparse solvers — add structured diagnostics
2. Neural operations — add performance warnings
3. Memory allocation — add allocation context
4. NCCL operations — add collective diagnostics

</codebase_context>

<specifics>

## Specific Ideas

- Add structured logging to krylov solver iterations
- Add memory allocation context to pool operations
- Add performance hints to slow operations
- Add NVTX ranges to critical paths
</specifics>

<deferred>

## Deferred Ideas

None — scope is well-defined

</deferred>

---

*Phase: 109-Error/Log Messages*
*Context gathered: 2026-05-07*
