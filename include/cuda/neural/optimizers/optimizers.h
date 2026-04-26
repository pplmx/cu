#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

namespace cuda::neural::optimizers {

struct OptimizerConfig {
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.01f;
    bool fused = true;
};

class AdamWOptimizer {
public:
    explicit AdamWOptimizer(const OptimizerConfig& config);
    ~AdamWOptimizer();

    void step(
        float* params,
        const float* grads,
        size_t num_elements,
        int step,
        cudaStream_t stream = nullptr
    );

    void set_learning_rate(float lr);
    void set_weight_decay(float wd);
    float get_learning_rate() const { return config_.learning_rate; }
    float get_weight_decay() const { return config_.weight_decay; }

    void zero_momentum();
    void zero_grad();

private:
    OptimizerConfig config_;
    std::vector<float> m_data_;
    std::vector<float> v_data_;
    bool initialized_ = false;
};

struct LAMBConfig {
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-6f;
    float weight_decay = 0.01f;
    float clamp_value = 10.0f;
    bool use_layer_adaptation = true;
};

class LAMBOptimizer {
public:
    explicit LAMBOptimizer(const LAMBConfig& config);
    ~LAMBOptimizer();

    void step(
        float* params,
        const float* grads,
        size_t num_elements,
        int step,
        float* layer_norm_1 = nullptr,
        float* layer_norm_2 = nullptr,
        cudaStream_t stream = nullptr
    );

    void set_learning_rate(float lr);
    float get_learning_rate() const { return config_.learning_rate; }

    void zero_momentum();
    void zero_grad();

private:
    LAMBConfig config_;
    std::vector<float> m_data_;
    std::vector<float> v_data_;
    bool initialized_ = false;
};

struct GradientClipConfig {
    float max_norm = 1.0f;
    enum class NormType { L2, Inf };
    NormType norm_type = NormType::L2;
};

float clip_gradients(
    float* grads,
    size_t num_elements,
    const GradientClipConfig& config,
    cudaStream_t stream = nullptr
);

float compute_gradient_norm(
    const float* grads,
    size_t num_elements,
    GradientClipConfig::NormType norm_type = GradientClipConfig::NormType::L2,
    cudaStream_t stream = nullptr
);

class GradientClipper {
public:
    explicit GradientClipper(const GradientClipConfig& config);

    float clip(float* grads, size_t num_elements, cudaStream_t stream = nullptr);
    float compute_norm(const float* grads, size_t num_elements, cudaStream_t stream = nullptr);

    void set_max_norm(float max_norm);
    float get_max_norm() const { return config_.max_norm; }

private:
    GradientClipConfig config_;
};

}  // namespace cuda::neural::optimizers
