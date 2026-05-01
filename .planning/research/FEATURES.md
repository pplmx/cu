# Features Research — Sparse Solver Acceleration

## Feature Categories

### Table Stakes (Expected)

| Feature | Description | Complexity |
|---------|-------------|------------|
| Jacobi Preconditioner | Diagonal scaling: M = diag(A)⁻¹ | Low |
| ILU(0) Preconditioner | Single-level incomplete LU with zero fill-in | Medium |
| RCM Ordering | Reverse Cuthill-McKee bandwidth reduction | Low-Medium |

### Differentiators (Nova-Specific)

| Feature | Description | Complexity |
|---------|-------------|------------|
| GPU-Accelerated Setup | Build preconditioner on GPU | Medium |
| Adaptive Drop Tolerance | ILU(k) with fill control | High |
| Reusable Workspace | Pool-based preconditioner storage | Low |

## Expected Behavior

### Jacobi Preconditioner
```
Input: SparseMatrix A, tolerance τ
Output: Vector d (reciprocal of diagonal)

For each iteration:
  x_new = D⁻¹ * (b - (L+U) * x_old)  // 1 matrix-vector multiply
```

**Convergence:** Slow for ill-conditioned matrices, but O(n) setup.

### ILU Preconditioner
```
Input: SparseMatrix A, drop tolerance τ
Output: L, U (incomplete factors)

Setup: O(nnz(A) * fill_level)
Apply: Forward solve + backward solve

// cuSPARSE csrilu0 performs ILU(0) in-place
cusparseMatDescr_t descrA;
cusparseCsrilu0(..., descrA, ..., reorder);
```

### RCM Ordering
```
Input: SparseMatrix A in natural ordering
Output: Permutation vector P

Benefit: Reduces bandwidth, improves ILU fill-in patterns
Typical improvement: 20-50% reduction in fill-in for structured grids
```

## Dependencies

- Jacobi: None (uses existing SpMV)
- ILU: cuSPARSE `csrilu0` / `csrilu02`
- RCM: Thrust graph algorithms or custom BFS-based implementation
