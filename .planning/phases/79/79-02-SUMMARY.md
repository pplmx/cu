---
phase: 79
plan: 02
type: summary
wave: 2
status: complete
depends_on: [79-00, 79-01]
files_modified:
  - include/cuda/sparse/sparse_ops.hpp
---

# Phase 79, Plan 02: SpMV Operations

## Status: Complete

### Implementation Details

**ELL SpMV Algorithm:**

```cpp
for each row i:
    sum = 0
    base = i * max_nnz_per_row
    for j = 0 to max_nnz_per_row:
        col = col_indices[base + j]
        if col >= 0:  // Skip padding
            sum += values[base + j] * x[col]
    y[i] = sum
```

**SELL SpMV Algorithm:**

```cpp
for each slice s:
    slice_base = slice_ptr[s]
    slice_nnz = (slice_ptr[s+1] - slice_ptr[s]) / slice_height
    for local_i in slice:
        global_i = slice_start + local_i
        sum = 0
        base = slice_base + local_i * slice_nnz
        for j = 0 to slice_nnz:
            col = col_indices[base + j]
            if col >= 0:
                sum += values[base + j] * x[col]
        y[global_i] = sum
```

### Free Function Wrappers

Added `sparse_mv()` overloads for ELL and SELL that delegate to `SparseOps::spmv`.

### Numerical Accuracy

Both implementations produce bit-exact results matching CSR baseline:

- y[0] = 7.0 (1×1 + 2×3)
- y[1] = 6.0 (3×2)
- y[2] = 19.0 (4×1 + 5×3)

---

## Summary generated: 2026-05-01
