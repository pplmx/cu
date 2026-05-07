# Phase 109: Error/Log Messages - Plan

**Phase:** 109-Error/Log Messages
**Status:** Planned
**Created:** 2026-05-07

## Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| SC-01 | All error messages include actionable guidance | All error types |
| SC-02 | Structured logging with ERROR/WARN/INFO/DEBUG/TRACE | 5 levels |
| SC-03 | Context included in error messages | device, stream, operation |
| SC-04 | Log macros support compile-time disable | All levels |
| SC-05 | DEBUG/TRACE options in performance paths | Key paths |
| SC-06 | Error codes support programmatic handling | All error types |
| SC-07 | NVTX ranges use descriptive names | All ranges |

## Task Breakdown

### T-01: Error Message Improvements
Priority: **High**

Files:
- `src/cuda/error/*.cpp`

**Actions:**
- Review all error messages for actionability
- Add "how to fix" guidance to all error types
- Include relevant context (device ID, stream, operation)

### T-02: Log Level Implementation
Priority: **High**

Files:
- `include/cuda/observability/logger.hpp` (create if missing)

**Actions:**
- Define log levels: ERROR, WARN, INFO, DEBUG, TRACE
- Create NOVA_LOG macro with level filtering
- Support compile-time disable via NDEBUG

### T-03: Solver Diagnostics
Priority: **Medium**

Files:
- `src/cuda/sparse/krylov.cpp`

**Actions:**
- Add iteration count logging at INFO level
- Add convergence warnings at WARN level
- Add detailed residual logging at DEBUG level

### T-04: Memory Allocation Context
Priority: **Medium**

Files:
- `src/cuda/memory/distributed_pool.cpp`

**Actions:**
- Add allocation size logging at INFO level
- Add device selection logging at DEBUG level
- Add memory pressure warnings at WARN level

## Execution Order

1. **T-01 (Error Messages):** Foundation - improve all error handling
2. **T-02 (Log Levels):** Create logging infrastructure
3. **T-03 (Solvers):** High-value diagnostics for sparse operations
4. **T-04 (Memory):** Memory allocation visibility

## Verification

1. Review error messages for guidance completeness
2. Verify log level filtering works
3. Check NVTX ranges have descriptive names

## Notes

- Follow existing error structure from Phase 107
- Preserve backward compatibility with existing error codes
- Make logging optional via compile flags

---

*Plan created: 2026-05-07*
*Tasks: 4 | Target files: ~5*
