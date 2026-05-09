# Phase 55: Linear Algebra Extras - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>

## Phase Boundary

Implement GPU-accelerated linear algebra extras using cuSOLVER. This includes SVD (singular value decomposition), symmetric eigenvalue decomposition, and matrix factorization methods (QR, Cholesky, LDL). All operations should integrate with the existing Buffer pattern.
</domain>

<decisions>

## Implementation Decisions

### Technology

- Use cuSOLVER (cusolverDn) for all decompositions
- Standard SVD via gesvdj (Jacobi), randomized via gesvdr for approximate PCA
- Eigenvalue decomposition via syevd for symmetric matrices
- Matrix factorizations via geqrf (QR), potrf (Cholesky), sytrf (LDL)

### API Design

- Follow existing cuda:: namespace conventions
- Create new cuda::linalg namespace for linear algebra operations
- Return decomposition results via struct with U, S, Vt components for SVD
- Handle workspace allocation internally (cusolver requires workspace buffers)

### Integration

- Reuse existing Buffer<T> for matrix storage
- Support batched operations where applicable
- Condition number estimation for numerical stability

</decisions>

## Existing Code Insights

### Reusable Assets

- cuda::memory::Buffer<T> for GPU memory
- cuda::detail::KernelLauncher for any custom kernels
- Existing CUDA error handling patterns

### Established Patterns

- Header-only declarations, .cu implementations
- Template-based generic algorithms
- CUDA_CHECK for error handling

### Integration Points

- include/cuda/linalg/ for new namespace
- CMake integration with CUDA::cusolver

</code_context>

<specifics>

## Specific Ideas

- SVD should support full and thin modes
- QR returns Q and R matrices
- Cholesky requires symmetric positive definite input
- Condition number computed from singular values

</specifics>

<deferred>

## Deferred Ideas

- Generalized SVD for rectangular matrices — Phase 59+
- Sparse eigensolver via cuDSS — future work

</deferred>
