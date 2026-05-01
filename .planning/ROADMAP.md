# Roadmap — v2.10 Sparse Solver Acceleration

## Milestone Summary

**Milestone:** v2.10 Sparse Solver Acceleration
**Goal:** Accelerate Krylov solver convergence with preconditioners and sparse matrix ordering
**Requirements:** 11 (PRECOND-01 to PRECOND-04, SOLVER-01 to SOLVER-03, TEST-01 to TEST-05)
**Phases:** 5

---

## Phase 88: Jacobi Preconditioner Foundation

**Goal:** Implement Jacobi preconditioner with weighted relaxation and unified Preconditioner interface

**Requirements:**
- PRECOND-01: JacobiPreconditioner with weighted diagonal scaling (ω)
- PRECOND-02: Preconditioner base interface
- TEST-01: Unit tests for Jacobi preconditioner

**Success Criteria:**
1. JacobiPreconditioner extracts diagonal from SparseMatrix
2. Weighted variant accepts ω parameter (0 < ω ≤ 2)
3. Preconditioner::apply() correctly scales vector
4. Zero diagonal detection throws descriptive error
5. Unit tests pass with >95% coverage

**Implementation Notes:**
- Use existing SpMV for matrix-vector multiply
- Store diagonal as Buffer<T> on GPU
- ω parameter in constructor: `JacobiPreconditioner(T omega = 1.0)`

---

## Phase 89: RCM Matrix Ordering

**Goal:** Implement RCM bandwidth reduction for improved ILU fill-in patterns

**Requirements:**
- PRECOND-03: RCM matrix ordering
- TEST-03: Unit tests for RCM ordering

**Success Criteria:**
1. BFS-based RCM correctly computes level sets
2. Permutation vector reduces bandwidth by ≥20% on test matrices
3. Matrix can be reordered in-place or copied
4. Inverse permutation correctly restores original ordering
5. Unit tests pass on known graph structures

**Implementation Notes:**
- BFS from multiple starting nodes (lowest-degree first)
- Level-by-level frontier tracking
- Bandwidth = max(|i - j|) for non-zero A[i,j]

---

## Phase 90: ILU Preconditioner

**Goal:** Implement ILU(0) incomplete factorization via cuSPARSE

**Requirements:**
- PRECOND-04: ILU(0) preconditioner via cuSPARSE
- TEST-02: Unit tests for ILU preconditioner

**Success Criteria:**
1. ILU setup completes via cusparseCsrilu0
2. Fill-in ratio (nnz(L+U)/nnz(A)) logged and monitored
3. Forward solve then backward solve for apply
4. Zero pivot handling with descriptive error
5. Unit tests pass on known matrices

**Implementation Notes:**
- Use CusparseContext singleton for handle management
- Separate workspace for setup (can be reused)
- Warning if fill-in ratio exceeds threshold

---

## Phase 91: Solver Integration

**Goal:** Wire preconditioners into existing CG, GMRESGPU, and BiCGSTAB solvers

**Requirements:**
- SOLVER-01: CG with preconditioner support
- SOLVER-02: GMRESGPU with preconditioner support
- SOLVER-03: BiCGSTAB with preconditioner support
- TEST-04: E2E convergence tests

**Success Criteria:**
1. ConjugateGradient::set_preconditioner() accepts any Preconditioner
2. GMRESGPU::set_preconditioner() accepts any Preconditioner
3. BiCGSTAB::set_preconditioner() accepts any Preconditioner
4. Left preconditioning applied in each iteration
5. Iteration count reduces on ill-conditioned test matrices

**Implementation Notes:**
- Template on preconditioner type for zero overhead
- Apply preconditioner: z = M⁻¹ * r
- Document preconditioning strategy in solver headers

---

## Phase 92: Performance & Integration

**Goal:** Validate performance improvements and finalize milestone

**Requirements:**
- TEST-05: Performance benchmarks
- Integration tests
- Documentation updates

**Success Criteria:**
1. Jacobi reduces CG iterations by ≥20% on test matrices
2. ILU reduces iterations by ≥50% on test matrices
3. Setup time documented for both preconditioners
4. Crossover point identified (when preconditioner helps)
5. Documentation updated with preconditioner examples

**Implementation Notes:**
- Use existing benchmark infrastructure from v1.7
- Compare: iterations, total time, setup time
- Test matrices: identity, diagonal, tridiagonal, Laplacian

---

## Phase Map

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 88 | Jacobi Foundation | Jacobi + interface | PRECOND-01, PRECOND-02, TEST-01 | 5 |
| 89 | RCM Ordering | Bandwidth reduction | PRECOND-03, TEST-03 | 5 |
| 90 | ILU Preconditioner | cuSPARSE ILU(0) | PRECOND-04, TEST-02 | 5 |
| 91 | Solver Integration | Wire into solvers | SOLVER-01, SOLVER-02, SOLVER-03, TEST-04 | 5 |
| 92 | Performance & Integration | Validate + document | TEST-05 | 5 |

---

## Traceability Matrix

| REQ | Phase 88 | Phase 89 | Phase 90 | Phase 91 | Phase 92 |
|-----|----------|----------|----------|----------|----------|
| PRECOND-01 | ✓ | | | | |
| PRECOND-02 | ✓ | | | | |
| PRECOND-03 | | ✓ | | | |
| PRECOND-04 | | | ✓ | | |
| SOLVER-01 | | | | ✓ | |
| SOLVER-02 | | | | ✓ | |
| SOLVER-03 | | | | ✓ | |
| TEST-01 | ✓ | | | | |
| TEST-02 | | | ✓ | | |
| TEST-03 | | ✓ | | | |
| TEST-04 | | | | ✓ | |
| TEST-05 | | | | | ✓ |

**Coverage:** 11/11 requirements mapped (100%)
