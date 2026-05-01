# Architecture Research — Sparse Solver Acceleration

## Integration with Existing Architecture

### Layer: API (unchanged)
```cpp
// User-facing API additions
namespace nova::sparse {
    // New preconditioner API
    template<typename T>
    class JacobiPreconditioner {
    public:
        explicit JacobiPreconditioner(T omega = 1.0);
        void setup(const SparseMatrix<T>& A);
        void apply(const Buffer<T>& r, Buffer<T>& z);
    };
    
    template<typename T>
    class ILUPreconditioner {
    public:
        ILUPreconditioner(int level = 0, T drop_tol = 0.0);
        void setup(const SparseMatrix<T>& A);
        void apply(const Buffer<T>& r, Buffer<T>& z);
    };
    
    // Solver configuration extension
    template<typename T>
    class ConjugateGradient {
    public:
        void set_preconditioner(std::unique_ptr<Preconditioner<T>> prec);
        // ...
    };
}
```

### Layer: Algorithm (extended)
New files:
- `include/cuda/sparse/preconditioner.hpp` — base + Jacobi + ILU
- `include/cuda/sparse/reordering.hpp` — RCM/MD algorithms
- `src/cuda/sparse/preconditioner.cu` — GPU implementations
- `src/cuda/sparse/reordering.cu` — ordering kernels

### Data Flow Changes

```
Before (v2.9):
  r = b - A*x
  p = r  (or β*p)
  x_new = x + α*p
  r_new = r - α*A*p

After (v2.10 with preconditioner):
  r = b - A*x
  z = M⁻¹ * r    // NEW: preconditioner apply
  p = z + β*p_old (or z if first iteration)
  x_new = x + α*p
  r_new = r - α*A*p
```

### Suggested Build Order

1. **Phase 88: Jacobi Preconditioner** — Simplest, immediate benefit
2. **Phase 89: RCM Ordering** — Infrastructure for ILU
3. **Phase 90: ILU Preconditioner** — Core feature, uses cuSPARSE
4. **Phase 91: Solver Integration** — Wire into CG/GMRES/BiCGSTAB
5. **Phase 92: Testing & Benchmarks** — E2E validation, performance
