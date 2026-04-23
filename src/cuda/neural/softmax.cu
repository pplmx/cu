#include "cuda/neural/softmax.h"

#include "cuda/device/error.h"

#include <algorithm>
#include <cmath>

namespace cuda::neural {

SoftmaxResult::SoftmaxResult(int outer_dim, int inner_dim)
    : outer_dim(outer_dim), inner_dim(inner_dim) {
    size = outer_dim * inner_dim;
    output = new float[size];
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
}

SoftmaxResult::~SoftmaxResult() {
    clear();
}

void SoftmaxResult::upload() {
    CUDA_CHECK(cudaMemcpy(d_output, output, size * sizeof(float), cudaMemcpyHostToDevice));
}

void SoftmaxResult::download() {
    CUDA_CHECK(cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void SoftmaxResult::clear() {
    delete[] output;
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
    output = nullptr;
}

namespace {

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int outer_dim,
    int inner_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dim * inner_dim) return;

    int outer_idx = idx / inner_dim;
    int inner_idx = idx % inner_dim;

    float max_val = -INFINITY;
    for (int j = 0; j < inner_dim; ++j) {
        max_val = fmaxf(max_val, input[outer_idx * inner_dim + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < inner_dim; ++j) {
        sum += expf(input[outer_idx * inner_dim + j] - max_val);
    }

    output[idx] = expf(input[idx] - max_val) / sum;
}

__global__ void softmax_stable_kernel(
    const float* input,
    float* output,
    int outer_dim,
    int inner_dim
) {
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * inner_dim;

    float thread_max = -INFINITY;
    for (int i = tid; i < inner_dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[block_offset + i]);
    }
    shared_max[tid] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    float max_val = shared_max[0];

    __shared__ float shared_sum[256];
    float thread_sum = 0.0f;
    for (int i = tid; i < inner_dim; i += blockDim.x) {
        thread_sum += expf(input[block_offset + i] - max_val);
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    float sum = shared_sum[0];

    for (int i = tid; i < inner_dim; i += blockDim.x) {
        output[block_offset + i] = expf(input[block_offset + i] - max_val) / sum;
    }
}

}  // anonymous namespace

void softmax(
    const float* input,
    float* output,
    int outer_dim,
    int inner_dim,
    cudaStream_t stream
) {
    int grid_size = outer_dim;
    int block_size = 256;

    softmax_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, outer_dim, inner_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

void softmax_stable(
    const float* input,
    float* output,
    int outer_dim,
    int inner_dim,
    cudaStream_t stream
) {
    int block_size = 256;
    int shared_mem = block_size * sizeof(float) * 2;

    softmax_stable_kernel<<<outer_dim, block_size, shared_mem, stream>>>(
        input, output, outer_dim, inner_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

void log_softmax(
    const float* input,
    float* output,
    int outer_dim,
    int inner_dim,
    cudaStream_t stream
) {
    int grid_size = outer_dim;
    int block_size = 256;

    softmax_stable_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, outer_dim, inner_dim
    );
    CUDA_CHECK(cudaGetLastError());

    output = output;
}

}  // namespace cuda::neural
