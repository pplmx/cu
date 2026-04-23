#include "cuda/neural/activations.h"

#include "cuda/device/error.h"

#include <cmath>

namespace cuda::neural {

namespace {

__global__ void relu_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_inplace_kernel(
    float* data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void leaky_relu_kernel(
    const float* input,
    float* output,
    int size,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : alpha * x;
    }
}

__global__ void leaky_relu_inplace_kernel(
    float* data,
    int size,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = x > 0.0f ? x : alpha * x;
    }
}

__global__ void sigmoid_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void tanh_kernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

}  // anonymous namespace

void relu(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void relu_inplace(
    float* data,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    relu_inplace_kernel<<<grid_size, block_size, 0, stream>>>(
        data, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void leaky_relu(
    const float* input,
    float* output,
    int size,
    float alpha,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size, alpha
    );
    CUDA_CHECK(cudaGetLastError());
}

void leaky_relu_inplace(
    float* data,
    int size,
    float alpha,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    leaky_relu_inplace_kernel<<<grid_size, block_size, 0, stream>>>(
        data, size, alpha
    );
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

void tanh_activation(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    tanh_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, size
    );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda::neural
