# Phase 108: Code Comments - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>

## Phase Boundary

Add inline code comments across all source files in the five-layer architecture. Comments should explain the "why" not the "what" - focusing on non-obvious logic, algorithm choices, performance considerations, and edge case handling.
</domain>

<decisions>

## Implementation Decisions

### Comment Style

- **C-01:** Use `//` for single-line comments, reserve `/** */` for Doxygen (already done in Phase 107)
- **C-02:** Comments precede code they explain (not inline)
- **C-03:** Explain reasoning, not implementation details

### Layer Priority

- **C-04:** Priority order: memory → device → algo → api → observability
- **C-05:** Start with highest-risk modules: neural/, inference/, sparse/, nccl/

### Comment Categories

- **C-06:** Algorithm rationale: Why this approach over alternatives
- **C-07:** Performance notes: Why this ordering/structure
- **C-08:** Edge case handling: Why this condition check exists
- **C-09:** Thread-safety: What invariants are protected
- **C-10:** Error handling: Why this specific error path

</decisions>

<canonical_refs>

## Canonical References

### Documentation Standards

- `.planning/codebase/CONVENTIONS.md` §Code Comments — Commenting style guide
- `.planning/codebase/ARCHITECTURE.md` §Five Layers — Layer descriptions

### Related Phases

- Phase 107 (API Documentation) — Doxygen comments (already complete)
- Phase 109 (Error/Log Messages) — Logging comment guidelines
</canonical_refs>

<codebase_context>

## Existing Code Insights

### Well-Commented Files (Reference)

- `include/cuda/error/cuda_error.hpp` — Good inline comment examples
- `src/cuda/error/*.cpp` — Implementation comments

### Layers to Comment

1. **cuda/memory/** (2 files) — Memory management, buffer allocation
2. **cuda/device/** (2 files) — Device management, CUDA context
3. **cuda/algo/** (2 files) — Algorithm implementations
4. **cuda/error/** (6 files) — Error handling (reference only, already good)
5. **cuda/observability/** (0 header files, need to check src/)

### High-Complexity Modules (Priority)

1. `src/cuda/neural/` — 13 files, complex neural network ops
2. `src/cuda/nccl/` — 11 files, distributed communication
3. `src/cuda/production/` — 8 files, production utilities
4. `src/cuda/sparse/` — 3 files, sparse matrix operations
5. `src/cuda/inference/` — 3 files, inference optimization

</codebase_context>

<specifics>

## Specific Ideas

- Add thread-safety comments to concurrent data structures
- Explain memory allocation strategies
- Document algorithm complexity assumptions
- Note CUDA kernel launch configuration rationale
</specifics>

<deferred>

## Deferred Ideas

None — scope is well-defined

</deferred>

---

*Phase: 108-Code Comments*
*Context gathered: 2026-05-07*
