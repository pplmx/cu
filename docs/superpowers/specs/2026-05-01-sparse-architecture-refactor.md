# v2.9 Architecture Refactor: Unified Sparse + cuSPARSE Integration

**Date:** 2026-05-01
**Author:** GSD Workflow
**Status:** Approved

## Context

The Nova CUDA library's sparse module (v2.8) has critical architectural issues:

1. **CPU-only implementations** — SpMV, SpMM, and Krylov solvers run on CPU via sequential loops
2. **Duplicate CSR abstractions** — `SparseMatrixCSR<T>` (sparse/) vs `CSRGraph` (graph/) represent the same data differently
3. **Inconsistent memory management** — Sparse module uses `std::vector` instead of `cuda::memory::Buffer<T>`
4. **Missing GPU acceleration** — Library is named "CUDA" but sparse operations are not GPU-accelerated

This refactor addresses these issues while maintaining backward compatibility.

## Design

### 1. Memory Layer Integration

All sparse matrix storage will use `cuda::memory::Buffer<T>` for automatic GPU memory management:

```cpp
template<typename T>
class SparseMatrix {
public:
    // Factory methods for construction
    static std::optional<SparseMatrix> FromDense(const T* dense, int rows, int cols);
    static SparseMatrix FromHostData(std::vector<T> values, std::vector<int> row_offsets,
                                      std::vector<int> col_indices, int rows, int cols);

    // Core properties
    int rows() const { return num_rows_; }
    int cols() const { return num_cols_; }
    int nnz() const { return values_.size(); }

    // Device pointers for GPU kernels
    T* values() { return values_.data(); }
    int* row_offsets() { return row_offsets_.data(); }
    int* col_indices() { return col_indices_.data(); }

    const T* values() const { return values_.data(); }
    const int* row_offsets() const { return row_offsets_.data(); }
    const int* col_indices() const { return col_indices_.data(); }

private:
    memory::Buffer<T> values_;
    memory::Buffer<int> row_offsets_;
    memory::Buffer<int> col_indices_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};
```

### 2. cuSPARSE Integration Layer

A singleton context wraps cuSPARSE handles:

```cpp
namespace cuda::sparse::detail {

class CusparseContext {
public:
    static CusparseContext& get();

    cusparseHandle_t handle() { return handle_; }

    // SpMV operations
    template<typename T>
    void spmv(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream = nullptr);

    // SpMM operations  
    template<typename T>
    void spmm(const SparseMatrix<T>& A, const T* B, T* C, int num_cols, cudaStream_t stream = nullptr);

private:
    CusparseContext();
    ~CusparseContext();

    cusparseHandle_t handle_;
};

}  // namespace cuda::sparse::detail
```

### 3. Unified CSR Interface

The new `SparseMatrix<T>` class serves both numerical computing and graph use cases:

```cpp
namespace cuda::sparse {

// Factory for graph construction
template<typename T>
SparseMatrix<T> FromEdgeList(const std::vector<std::pair<int,int>>& edges,
                              int num_vertices,
                              const std::vector<T>* weights = nullptr);

// GPU-accelerated operations
template<typename T>
void spmv(const SparseMatrix<T>& A, const T* x, T* y);

template<typename T>
void spmm(const SparseMatrix<T>& A, const T* B, T* C, int num_cols);

}  // namespace cuda::sparse
```

### 4. GPU-Accelerated Krylov Solvers

Solvers use cuSOLVER sparse solvers under the hood:

```cpp
namespace cuda::sparse {

template<typename T>
struct SolverResult {
    bool converged = false;
    int iterations = 0;
    T residual_norm = T{0};
    T relative_residual = T{0};
    SolverError error_code = SolverError::SUCCESS;
    std::vector<T> residual_history;
};

template<typename T>
class ConjugateGradient {
public:
    explicit ConjugateGradient(const SolverConfig<T>& config = {});

    SolverResult<T> solve(const SparseMatrix<T>& A, const T* b, T* x);

private:
    SolverConfig<T> config_;
    memory::Buffer<T> workspace_;
};

template<typename T>
class GMRES {
public:
    explicit GMRES(const SolverConfig<T>& config = {}, int restart = 50);

    SolverResult<T> solve(const SparseMatrix<T>& A, const T* b, T* x);

private:
    SolverConfig<T> config_;
    int restart_;
    memory::Buffer<T> workspace_;
};

}  // namespace cuda::sparse
```

### 5. Format-Specific Implementations

Format-specific storage continues to exist for formats not fully supported by cuSPARSE:

```
include/cuda/sparse/
├── matrix.hpp           # Unified SparseMatrix<T> base
├── cusparse_context.hpp # cuSPARSE handle wrapper
├── krylov.hpp           # GPU-accelerated solvers
├── roofline.hpp         # Existing Roofline analysis
├── formats/
│   ├── csr.hpp          # CSR storage (uses cuSPARSE)
│   ├── csc.hpp          # CSC storage (uses cuSPARSE)
│   ├── ell.hpp          # ELL storage (custom kernels needed)
│   ├── sell.hpp         # SELL storage (custom kernels needed)
│   └── hyb.hpp          # HYB storage (custom kernels needed)
└── ops/
    ├── spmv.hpp         # Unified SpMV interface
    └── spmm.hpp         # Unified SpMM interface
```

### 6. Backward Compatibility

The old `SparseMatrixCSR<T>` is deprecated but remains functional:

```cpp
// Old API (deprecated)
template<typename T>
class [[deprecated("Use cuda::sparse::SparseMatrix<T> instead")]] 
SparseMatrixCSR {
    std::vector<T> values_;
    // ...
};
```

Migration path:
1. Old code continues to work (with deprecation warnings)
2. New code uses `cuda::sparse::SparseMatrix<T>`
3. Tests updated to verify both paths work

## Module Structure

### New Files

| File | Purpose |
|------|---------|
| `include/cuda/sparse/matrix.hpp` | Unified SparseMatrix class with Buffer<T> |
| `include/cuda/sparse/cusparse_context.hpp` | cuSPARSE handle management |
| `include/cuda/sparse/formats/csr.hpp` | CSR format (cuSPARSE-backed) |
| `include/cuda/sparse/formats/csc.hpp` | CSC format (cuSPARSE-backed) |
| `src/cuda/sparse/sparse_ops.cu` | GPU kernels for operations |
| `src/cuda/sparse/krylov_gpu.cpp` | cuSOLVER-based solvers |

### Modified Files

| File | Changes |
|------|---------|
| `include/cuda/sparse/krylov.hpp` | Add GPU solver implementations |
| `include/cuda/graph/csr_graph.h` | Deprecate, add conversion to SparseMatrix |
| `tests/sparse/*` | Add GPU tests, keep CPU tests for validation |

### Deleted Files

None — all existing APIs are deprecated, not removed.

## Dependencies

- cuSPARSE (NVIDIA) — for SpMV, SpMM on CSR/CSC
- cuSOLVER (NVIDIA) — for sparse solver routines
- Existing `cuda::memory::Buffer<T>` — for GPU memory management

## Constraints

1. **Backward compatibility** — Existing API must continue to work
2. **No breaking changes** — All v2.8 features must remain functional
3. **Performance** — New GPU implementations must be faster than CPU equivalents
4. **Memory efficiency** — Reuse workspace buffers across operations

## Success Criteria

1. All existing sparse tests pass with GPU-accelerated implementation
2. New `SparseMatrix<T>` class uses `cuda::memory::Buffer<T>`
3. cuSPARSE integrated for CSR/CSC SpMV and SpMM
4. Krylov solvers use GPU-accelerated operations
5. ELL/SELL/HYB formats continue to work (may use custom kernels)
6. Conversion between `CSRGraph` and `SparseMatrix<T>` works
7. Backward-compatible `SparseMatrixCSR<T>` marked deprecated

## Out of Scope

- Implementing custom SpMV kernels for ELL/SELL/HYB (future work)
- Multi-GPU sparse operations (depends on this refactor)
- Sparse format autotuning based on Roofline analysis (future work)

## Traceability

| Requirement | Phase | Notes |
|-------------|-------|-------|
| Unified SparseMatrix with Buffer<T> | TBD | Foundation |
| cuSPARSE integration | TBD | CSR/CSC support |
| GPU-accelerated Krylov | TBD | Depends on cuSPARSE |
| Backward compatibility | TBD | Deprecation path |
| Test coverage | TBD | GPU + validation tests |

---

*Spec created: 2026-05-01*
*v2.9: Architecture Refactor*
