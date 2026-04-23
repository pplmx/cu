#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace cuda::neural {

struct LayerNormResult {
    float* output;
    float* mean;
    float* variance;
    float* d_output;
    float* d_mean;
    float* d_variance;
    int size;

    LayerNormResult() : output(nullptr), mean(nullptr), variance(nullptr),
                        d_output(nullptr), d_mean(nullptr), d_variance(nullptr), size(0) {}
    explicit LayerNormResult(int size);
    ~LayerNormResult();

    void upload();
    void download();
    void clear();
};

struct LayerNormParams {
    int normalized_shape;
    float eps = 1e-5f;
    bool elementwise_affine = true;
};

void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean,
    float* variance,
    int batch_size,
    int normalized_shape,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr
);

void layer_norm_inference(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int normalized_shape,
    float eps = 1e-5f,
    cudaStream_t stream = nullptr
);

}  // namespace cuda::neural
