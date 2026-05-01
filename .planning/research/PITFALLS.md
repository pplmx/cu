# Pitfalls Research — Sparse Solver Acceleration

## Common Mistakes When Adding Preconditioners

### 1. Diagonal Zero Detection
**Problem:** Jacobi fails when matrix has zero diagonal entries.
**Prevention:** Validate diagonal before setup; throw descriptive error.
```cpp
// Check for zeros
if (abs(diag[i]) < epsilon) {
    throw sparse_error("Zero diagonal at row {}", i);
}
```

### 2. ILU Fill-In Explosion
**Problem:** Without drop tolerance, ILU(0) can produce excessive fill-in.
**Prevention:** 
- Monitor nnz(L+U) / nnz(A) ratio
- Warn if fill-in exceeds threshold (e.g., 5x original)

### 3. GPU Memory for ILU Workspace
**Problem:** ILU setup requires workspace proportional to matrix size.
**Prevention:**
- Estimate workspace size before allocation
- Provide graceful fallback with clear error message

### 4. Preconditioner Reuse
**Problem:** Recreating preconditioner each iteration wastes setup cost.
**Prevention:**
- Separate `setup()` from `apply()` phases
- Document setup cost vs iteration savings

### 5. Synchronization Overhead
**Problem:** Preconditioner apply adds kernel launches.
**Prevention:**
- Batch preconditioner operations where possible
- Profile to verify net benefit

## Phase-Specific Warnings

### Phase 88 (Jacobi)
- Ensure thread-safety for shared preconditioners
- Validate on symmetric positive definite matrices

### Phase 89 (RCM)
- BFS-based RCM requires level-set frontier tracking
- Memory for BFS queue scales with matrix bandwidth

### Phase 90 (ILU)
- cuSPARSE ILU requires specific matrix format (CSR)
- Handle cusparseStatus_zero_pivot gracefully

### Phase 91 (Integration)
- Left vs right preconditioning semantics differ
- Document which is used and why

### Phase 92 (Testing)
- Compare iteration count with/without preconditioner
- Measure setup time vs solve time tradeoff
