#include "cuda/algo/flash_attention.h"

namespace cuda::algo {

std::unique_ptr<FlashAttention> create_flash_attention(
    const FlashAttentionConfig& config
) {
    return std::make_unique<FlashAttention>(config);
}

FlashAttention::FlashAttention(const FlashAttentionConfig& config)
    : config_(config) {}

FlashAttention::~FlashAttention() = default;

size_t FlashAttention::get_workspace_size() const {
    return 0;
}

void FlashAttention::ensure_workspace(size_t size) {
}

void FlashAttention::set_dropout(float rate, unsigned long seed) {
}

void FlashAttention::forward(
    memory::Buffer<float>& output,
    memory::Buffer<float>& attention_scores,
    const memory::Buffer<float>& query,
    const memory::Buffer<float>& key,
    const memory::Buffer<float>& value,
    const stream::Stream& stream
) {
    cudaStream_t s = stream.get();
}

void FlashAttention::forward_bf16(
    memory::Buffer<void>& output,
    memory::Buffer<float>& attention_scores,
    const memory::Buffer<void>& query,
    const memory::Buffer<void>& key,
    const memory::Buffer<void>& value,
    const stream::Stream& stream
) {
}

void FlashAttention::backward(
    memory::Buffer<float>& d_query,
    memory::Buffer<float>& d_key,
    memory::Buffer<float>& d_value,
    const memory::Buffer<float>& d_output,
    const memory::Buffer<float>& attention_scores,
    const memory::Buffer<float>& query,
    const memory::Buffer<float>& key,
    const memory::Buffer<float>& value,
    const memory::Buffer<float>& output,
    const stream::Stream& stream
) {
}

}  // namespace cuda::algo
