#pragma once

#include <cuda_runtime.h>

namespace cuda::neural {

struct ActivationOptions {
    float alpha = 0.01f;
    float negative_slope = 0.01f;
};

void relu(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

void relu_inplace(
    float* data,
    int size,
    cudaStream_t stream = nullptr
);

void leaky_relu(
    const float* input,
    float* output,
    int size,
    float alpha = 0.01f,
    cudaStream_t stream = nullptr
);

void leaky_relu_inplace(
    float* data,
    int size,
    float alpha = 0.01f,
    cudaStream_t stream = nullptr
);

void sigmoid(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

void tanh_activation(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

}  // namespace cuda::neural
