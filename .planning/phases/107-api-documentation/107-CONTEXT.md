# Phase 107: API Documentation - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>

## Phase Boundary

Complete Doxygen documentation coverage for all public headers in `include/cuda/`. This includes adding @brief, @param/@return, @tparam, @defgroup modules, @deprecated, @code/@endcode examples, @see cross-references, and @note performance annotations.
</domain>

<decisions>

## Implementation Decisions

### Documentation Scope and Priority

- **D-01:** All public headers in `include/cuda/**/*.hpp` get documented — comprehensive coverage across the entire library
- **D-02:** Priority order follows five-layer architecture: memory → device → algo → api → other modules

### Doxygen Group Organization

- **D-03:** Use existing five-layer architecture groups from Phase 37 (memory, device, algo, api, observability)
- **D-04:** Extend groups as needed for new modules: sparse, quantize, gnn, inference, distributed, error, performance

### Example Code Strategy

- **D-05:** One representative example per module — per-layer entry points showing basic usage
- **D-06:** Example modules: memory (Buffer<T>), device (CUDA_CHECK), algo (reduce_sum/FlashAttention), inference (Scheduler), sparse (SparseMatrix), distributed (NcclContext)
- **D-07:** Existing `examples/` directory provides comprehensive runnable examples — Doxygen examples are illustrative, not complete

### Performance Notes Coverage

- **D-08:** Add @note performance annotations to compute-intensive APIs only
- **D-09:** Covered modules: `include/cuda/algo/`, `include/cuda/linalg/`, `include/cuda/neural/`, `include/cuda/sparse/` (solvers), `include/cuda/nccl/`, `include/cuda/inference/`
- **D-10:** Omit performance notes from simple getters/setters, RAII constructors/destructors, type conversions
- **D-11:** Performance notes content: time complexity, memory allocation requirements, GPU resource usage, known limitations

### Deprecation Handling

- **D-12:** Existing `[[deprecated]]` attributes should have corresponding @deprecated Doxygen tags with migration guidance (e.g., SparseMatrixCSR → SparseMatrix<T>)
- **D-13:** Example: `@deprecated Use cuda::sparse::SparseMatrix<T> instead via ToSparseMatrix()`

### Cross-Reference Strategy

- **D-14:** Use @see to link related functions within the same module
- **D-15:** Cross-references connect: Buffer → MemoryPool, Scheduler → BlockManager → SequenceManager, SparseMatrix → SparseMatrixCSR (deprecated)

</decisions>

<canonical_refs>

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Documentation Standards

- `.planning/codebase/CONVENTIONS.md` §Documentation Standards — Doxygen style guide and examples
- `Doxyfile` — Existing Doxygen configuration

### Existing Documentation Work

- `.planning/phases/37-api-reference/CONTEXT.md` — Phase 37 context establishing initial @defgroup structure
- `.planning/phases/37-api-reference/` — Contains initial API documentation plan

### Architecture Context

- `.planning/codebase/ARCHITECTURE.md` — Five-layer architecture overview
- `.planning/codebase/STRUCTURE.md` — Codebase directory layout and file organization

### Requirements

- `.planning/ROADMAP.md` §Phase 107 — Phase goal and success criteria
- `.planning/REQUIREMENTS.md` §API Documentation — APIDOC-01 through APIDOC-08 requirements

</canonical_refs>

<codebase_context>

## Existing Code Insights

### Reusable Assets

- **Well-documented headers:** `include/cuda/error/cuda_error.hpp` — serves as template for Doxygen documentation style
- **Doxyfile:** Already configured with proper settings for HTML output, group definitions, warning levels
- **Existing @defgroup structure:** Phase 37 established memory, device, algo, api groups

### Established Patterns

- **Doxygen style:** Block comments with `@brief`, `@param`, `@return`, `@tparam`, `@note`, `@see`, `@code/@endcode`
- **File-level docs:** Consistent `@file`, `@brief`, `@defgroup @ingroup` pattern
- **Template documentation:** `@tparam` for template parameters with constraint descriptions

### Integration Points

- **Doxygen configuration:** `Doxyfile` controls HTML output directory (`docs/api`), input paths, and grouping
- **Header organization:** Files in `include/cuda/` subdirectories map to @defgroup modules
- **CMake integration:** Documentation build targets may need updates for CI integration (APIDOC-09)

</codebase_context>

<specifics>

## Specific Ideas

- Use `include/cuda/error/cuda_error.hpp` as reference implementation for documentation style
- Performance note example format: `@note Time complexity: O(n log n). Allocates temporary buffer of size 2*N.`
- Deprecation example: `@deprecated Use cuda::sparse::SparseMatrix<T> instead. See ToSparseMatrix() for migration.`
</specifics>

<deferred>

## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 107-API Documentation*
*Context gathered: 2026-05-07*
