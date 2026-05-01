# Research Summary — v2.10 Sparse Solver Acceleration

## Stack Additions

| Component | Source | Notes |
|-----------|--------|-------|
| Preconditioner interfaces | New | `preconditioner.hpp` |
| Jacobi implementation | Custom GPU | Uses existing SpMV |
| ILU implementation | cuSPARSE | `csrilu0` / `csrilu02` |
| RCM reordering | Custom GPU | BFS-based algorithm |
| Thrust/CUB | Bundled with CUDA | Reuse existing |

**No new external dependencies.**

## Feature Table Stakes

| Feature | Complexity | Notes |
|---------|------------|-------|
| Jacobi Preconditioner | Low | Diagonal scaling, O(n) setup |
| RCM Ordering | Low-Medium | Reduces bandwidth 20-50% |
| ILU(0) Preconditioner | Medium | cuSPARSE integration |

## Watch Out For

1. **Zero diagonal detection** — Validate before Jacobi setup
2. **ILU fill-in explosion** — Monitor nnz ratio, add drop tolerance
3. **GPU memory for workspace** — Estimate before allocation
4. **Setup vs solve tradeoff** — Profile to verify net benefit
5. **Left vs right preconditioning** — Document semantics

## Suggested Phases

| Phase | Feature | Rationale |
|-------|---------|-----------|
| 88 | Jacobi Preconditioner | Simplest, immediate benefit |
| 89 | RCM Ordering | Infrastructure for ILU |
| 90 | ILU Preconditioner | Core feature, cuSPARSE |
| 91 | Solver Integration | Wire into existing solvers |
| 92 | Testing & Benchmarks | E2E validation |

## Key Integration Points

- Extend `SparseMatrix<T>` with diagonal extraction
- Add `Preconditioner` interface to `solver.hpp`
- Configure solvers via `set_preconditioner()` method
- Reuse existing `Buffer<T>` and `MemoryPool` infrastructure
