#include "cuda/device/cublas_context.h"
#include "cuda/memory/buffer.h"
#include "cuda/memory/buffer-inl.h"

namespace cuda::algo {

// Matrix multiplication C = A * B using cuBLAS
// Matrix dimensions: A(mxk), B(kxn), C(mxn)
// Uses CUBLAS_OP_N for both inputs (no transpose)
// alpha=1, beta=0 means C = 1*A*B + 0*C = A*B (overwrites C)
void matrixMultiply(const memory::Buffer<float>& a, const memory::Buffer<float>& b, memory::Buffer<float>& c, int m, int n, int k) {
    // Create cuBLAS context - handles initialization and cleanup via RAII
    device::CublasContext ctx;

    // SGEMM parameters: C = alpha * A * B + beta * C
    // With alpha=1, beta=0: C = A * B (result overwrites existing C values)
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major memory layout
    // Leading dimension (lda, ldb, ldc) is stride between consecutive columns
    // For row-major storage accessed as column-major: lda=k, ldb=n, ldc=n
    CUBLAS_CHECK(cublasSgemm(ctx.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.data(), k, b.data(), n, &beta, c.data(), n));
}

// Double-precision matrix multiplication - same logic as float version above
// Uses cublasDgemm for double precision arithmetic
void matrixMultiply(const memory::Buffer<double>& a, const memory::Buffer<double>& b, memory::Buffer<double>& c, int m, int n, int k) {
    device::CublasContext ctx;

    const double alpha = 1.0;
    const double beta = 0.0;

    CUBLAS_CHECK(cublasDgemm(ctx.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.data(), k, b.data(), n, &beta, c.data(), n));
}

}  // namespace cuda::algo
