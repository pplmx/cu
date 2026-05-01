# Stack Research — Sparse Solver Acceleration

## Technology Stack Additions

### Libraries Required

| Library | Version | Purpose |
|---------|---------|---------|
| cuSOLVER | 20+ | ILU factorization via `cusolverDn<t>ilu0()` / `csrilu0()` |
| Thrust | (bundled) | Sparse matrix reordering algorithms |
| CUB | (bundled) | Parallel primitives for reordering |

### No New Dependencies

- cuSPARSE (already integrated in v2.9) — reused for SpMV in preconditioner apply
- Existing Buffer<T> infrastructure — reused for preconditioner storage
- Existing Krylov solver infrastructure — extended with preconditioner hooks

### Integration Points

```cpp
// Preconditioner interface (new)
namespace nova::sparse {
    class Preconditioner {
    public:
        virtual void apply(const Buffer<T>& in, Buffer<T>& out) = 0;
        virtual void setup(const SparseMatrix<T>& A) = 0;
    };
    
    class JacobiPreconditioner : public Preconditioner { ... };
    class ILUPreconditioner : public Preconditioner { ... };
}

// Solver extension
template<typename Solver, typename Preconditioner>
class PreconditionedSolver {
    void set_preconditioner(std::unique_ptr<Preconditioner> prec);
};
```

### What's NOT Being Added

- SuperLU / MUMPS integration — CPU-based, out of scope
- AMG (Algebraic Multigrid) — too complex, future milestone
- Domain decomposition — future work
