---
phase: 79
plan: 00
type: summary
wave: 0
status: complete
files_modified:
  - include/cuda/sparse/sparse_matrix.hpp
  - include/cuda/sparse/sparse_ops.hpp
---

# Phase 79, Plan 00: Interface Contracts

## Status: Complete

### Files Modified

1. **include/cuda/sparse/sparse_matrix.hpp**
   - Added `ELL` and `SELL` to `SparseFormat` enum
   - Added `SparseMatrixELL<T>` class declaration with all public methods
   - Added `SparseMatrixSELL<T>` class declaration with all public methods
   - Implemented `FromCSR` factory methods for both classes
   - Implemented accessor methods

2. **include/cuda/sparse/sparse_ops.hpp**
   - Added `spmv` overload for `SparseMatrixELL<T>`
   - Added `spmv` overload for `SparseMatrixSELL<T>`
   - Added `sparse_mv` free function overloads

### Interfaces Declared

**SparseMatrixELL<T>:**
- `static FromCSR(const SparseMatrixCSR<T>&)` — Convert CSR to ELL
- `int num_rows() const` / `int num_cols() const`
- `int nnz() const` / `int padded_nnz() const`
- `int max_nnz_per_row() const`
- `const T* values() const` / `const int* col_indices() const`
- `const int* row_offsets() const`

**SparseMatrixSELL<T>:**
- `static FromCSR(const SparseMatrixCSR<T>&, int slice_height=32)`
- `int num_rows() const` / `int num_cols() const`
- `int nnz() const` / `int padded_nnz() const`
- `int slice_height() const`
- `const T* values() const` / `const int* col_indices() const`
- `const int* slice_ptr() const`

**SparseOps<T>:**
- `static void spmv(const SparseMatrixELL<T>&, const T* x, T* y)`
- `static void spmv(const SparseMatrixSELL<T>&, const T* x, T* y)`

---
*Summary generated: 2026-05-01*
