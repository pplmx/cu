#include "cuda/neural/matmul.h"

#include "cuda/device/error.h"

#include <stdexcept>

namespace cuda::neural {

namespace {
cublasHandle_t g_cublas_handle = nullptr;
}

cublasHandle_t get_cublas_handle() {
    if (!g_cublas_handle) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
    }
    return g_cublas_handle;
}

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k,
    MatmulOptions options
) {
    cublasHandle_t handle = options.handle ? options.handle : get_cublas_handle();

    float alpha = options.alpha;
    float beta = options.beta;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        options.trans_a,
        options.trans_b,
        n,
        m,
        k,
        &alpha,
        B,
        n,
        A,
        k,
        &beta,
        C,
        n
    ));
}

void matmul_batch(
    const float* A,
    const float* B,
    float* C,
    int batch_count,
    int m,
    int n,
    int k,
    MatmulOptions options
) {
    cublasHandle_t handle = options.handle ? options.handle : get_cublas_handle();

    int lda = options.trans_a == CUBLAS_OP_N ? k : m;
    int ldb = options.trans_b == CUBLAS_OP_N ? n : k;
    int ldc = n;

    float alpha = options.alpha;
    float beta = options.beta;

    CUBLAS_CHECK(cublasSgemmBatched(
        handle,
        options.trans_a,
        options.trans_b,
        n,
        m,
        k,
        &alpha,
        &B,
        ldb,
        &A,
        lda,
        &beta,
        &C,
        ldc,
        batch_count
    ));
}

}  // namespace cuda::neural
