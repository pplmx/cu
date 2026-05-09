# Phase 107: API Documentation - Plan

**Phase:** 107-API Documentation
**Status:** Planned
**Created:** 2026-05-07

## Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| SC-01 | All include/cuda/**/*.hpp headers have @brief descriptions | 30/30 |
| SC-02 | All public function signatures have @param/@return documentation | 100% |
| SC-03 | Template specializations have @tparam documentation | All templates |
| SC-04 | Headers organized into @defgroup modules | 5 groups minimum |
| SC-05 | Deprecated functions have @deprecated with guidance | All deprecated |
| SC-06 | Key APIs have @code/@endcode examples | 1 per module |
| SC-07 | Related functions linked via @see | All linked |
| SC-08 | Performance notes added to critical paths | algo/sparse/nccl/inference |

## Task Breakdown

### T-01: Error Module (6 headers)

Files:

- `include/cuda/error/cuda_error.hpp` — already documented ✓
- `include/cuda/error/cublas_error.hpp`
- `include/cuda/error/timeout.hpp`
- `include/cuda/error/timeout_context.hpp`
- `include/cuda/error/degrade.hpp`
- `include/cuda/error/retry.hpp`

**Actions:**

- Add @file documentation with @defgroup error
- Document public functions with @param/@return
- Add @see cross-references between error types
- Add examples for error handling patterns

### T-02: Sparse Module (11 headers)

Files:

- `include/cuda/sparse/sparse_matrix.hpp`
- `include/cuda/sparse/matrix.hpp`
- `include/cuda/sparse/sparse_ops.hpp`
- `include/cuda/sparse/solver_workspace.hpp`
- `include/cuda/sparse/krylov.hpp`
- `include/cuda/sparse/reordering.hpp`
- `include/cuda/sparse/preconditioner.hpp`
- `include/cuda/sparse/hyb_matrix.hpp`
- `include/cuda/sparse/cusparse_context.hpp`
- `include/cuda/sparse/nvtx_sparse.hpp`
- `include/cuda/sparse/roofline.hpp`

**Actions:**

- Add @defgroup sparse
- Document all SparseMatrix format classes (CSR, CSC, ELL, SELL, HYB)
- Add @deprecated with migration guidance for SparseMatrixCSR
- Add @code examples for basic sparse operations
- Add @see references between matrix formats
- Add @note performance annotations for solvers (krylov.hpp)

### T-03: Quantize Module (10 headers)

Files:

- `include/cuda/quantize/quantize_tensor.hpp`
- `include/cuda/quantize/int8_kernels.hpp`
- `include/cuda/quantize/fp8_kernels.hpp`
- `include/cuda/quantize/fp8_gemm.hpp`
- `include/cuda/quantize/fp8_activation.hpp`
- `include/cuda/quantize/fp8_types.hpp`
- `include/cuda/quantize/quantize_ops.hpp`
- `include/cuda/quantize/calibrator.hpp`
- `include/cuda/quantize/benchmark.hpp`
- `include/cuda/quantize/qat.hpp`

**Actions:**

- Add @defgroup quantize
- Document quantization types (int8, fp8)
- Document calibration API
- Add @code examples for quantization workflow
- Add @see links to related operations

### T-04: GNN Module (3 headers)

Files:

- `include/cuda/gnn/sampling.hpp`
- `include/cuda/gnn/message_passing.hpp`
- `include/cuda/gnn/attention.hpp`

**Actions:**

- Add @defgroup gnn
- Document sampling strategies
- Document message passing API
- Add @note performance notes for graph operations

## Execution Order

1. **T-01 (error):** Reference implementation, already documented — review completeness
2. **T-02 (sparse):** Highest complexity, 11 headers — document first
3. **T-03 (quantize):** Medium complexity, 10 headers — document second
4. **T-04 (gnn):** Lowest complexity, 3 headers — document last

## Verification

After each task:

1. Run `doxygen Doxyfile 2>&1 | grep -c warning` — target: 0 warnings
2. Verify @brief on all documented headers
3. Check @defgroup membership via generated HTML

## Notes

- Use `include/cuda/error/cuda_error.hpp` as reference implementation
- Follow existing @defgroup structure from Phase 37
- Performance notes only on compute-intensive APIs (no simple getters)
- Deprecation migration guides include target API reference

---

*Plan created: 2026-05-07*
*Tasks: 4 | Estimated headers: 30*
