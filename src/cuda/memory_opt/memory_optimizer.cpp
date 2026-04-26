#include "cuda/memory_opt/memory_optimizer.h"

#include <algorithm>
#include <cstring>

#if NOVA_USE_ZSTD
#include <zstd.h>
#elif NOVA_USE_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif

namespace cuda::memory_opt {

CheckpointCompressor& CheckpointCompressor::instance() {
    static CheckpointCompressor compressor;
    return compressor;
}

void CheckpointCompressor::set_config(const CompressionConfig& config) {
    config_ = config;
}

CompressionConfig CheckpointCompressor::get_config() const {
    return config_;
}

size_t CheckpointCompressor::compress(
    const void* input,
    size_t input_size,
    void* output,
    size_t output_capacity
) {
    if (!config_.enable_compression || input_size < config_.min_size_for_compression) {
        if (input_size <= output_capacity) {
            memcpy(output, input, input_size);
            return input_size;
        }
        return 0;
    }

#if NOVA_USE_ZSTD
    size_t compressed_size = ZSTD_compress(
        output, output_capacity, input, input_size, config_.compression_level);
    if (ZSTD_isError(compressed_size)) {
        memcpy(output, input, std::min(input_size, output_capacity));
        return std::min(input_size, output_capacity);
    }
    return compressed_size;
#elif NOVA_USE_LZ4
    int compressed_size = LZ4_compress_default(
        static_cast<const char*>(input),
        static_cast<char*>(output),
        static_cast<int>(input_size),
        static_cast<int>(output_capacity));
    if (compressed_size == 0) {
        memcpy(output, input, std::min(input_size, output_capacity));
        return std::min(input_size, output_capacity);
    }
    return compressed_size;
#else
    memcpy(output, input, std::min(input_size, output_capacity));
    return std::min(input_size, output_capacity);
#endif
}

size_t CheckpointCompressor::decompress(
    const void* input,
    size_t input_size,
    void* output,
    size_t output_capacity
) {
#if NOVA_USE_ZSTD
    size_t decompressed_size = ZSTD_decompress(
        output, output_capacity, input, input_size);
    if (ZSTD_isError(decompressed_size)) {
        memcpy(output, input, std::min(input_size, output_capacity));
        return std::min(input_size, output_capacity);
    }
    return decompressed_size;
#elif NOVA_USE_LZ4
    int decompressed_size = LZ4_decompress_safe(
        static_cast<const char*>(input),
        static_cast<char*>(output),
        static_cast<int>(input_size),
        static_cast<int>(output_capacity));
    if (decompressed_size < 0) {
        memcpy(output, input, std::min(input_size, output_capacity));
        return std::min(input_size, output_capacity);
    }
    return decompressed_size;
#else
    memcpy(output, input, std::min(input_size, output_capacity));
    return std::min(input_size, output_capacity);
#endif
}

GradientAccumulator::GradientAccumulator(int max_accumulation_steps)
    : max_accumulation_steps_(max_accumulation_steps),
      accumulated_gradients_(0) {}

GradientAccumulator::~GradientAccumulator() {}

void GradientAccumulator::add_gradient(int step, const float* gradient, size_t size) {
    if (step != current_step_) {
        reset();
        current_step_ = step;
        accumulated_gradients_.resize(size, 0.0f);
        has_gradients_ = true;
    }

    if (accumulated_gradients_.size() != size) {
        accumulated_gradients_.resize(size, 0.0f);
    }

    for (size_t i = 0; i < size; ++i) {
        accumulated_gradients_[i] += gradient[i];
    }
}

void GradientAccumulator::get_accumulated_gradient(float* output) {
    if (!has_gradients_) return;
    memcpy(output, accumulated_gradients_.data(), accumulated_gradients_.size() * sizeof(float));
}

void GradientAccumulator::reset() {
    current_step_ = 0;
    has_gradients_ = false;
    accumulated_gradients_.clear();
}

bool GradientAccumulator::is_ready_to_apply() const {
    return has_gradients_ && current_step_ >= max_accumulation_steps_ - 1;
}

MemoryDefragmenter::MemoryDefragmenter(int device_id)
    : device_id_(device_id) {}

void MemoryDefragmenter::register_allocation(void* ptr, size_t size) {
    MemoryBlock block;
    block.ptr = ptr;
    block.size = size;
    block.in_use = true;
    block.device_id = device_id_;
    blocks_.push_back(block);
}

void MemoryDefragmenter::unregister_allocation(void* ptr) {
    for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
        if (it->ptr == ptr) {
            blocks_.erase(it);
            break;
        }
    }
}

void MemoryDefragmenter::defragment() {
    std::sort(blocks_.begin(), blocks_.end(),
        [](const MemoryBlock& a, const MemoryBlock& b) {
            return reinterpret_cast<size_t>(a.ptr) < reinterpret_cast<size_t>(b.ptr);
        });

    for (auto& block : blocks_) {
        if (!block.in_use || !reallocate_callback_) continue;

        void* new_ptr = block.ptr;
        size_t new_size = block.size;
        reallocate_callback_(block.ptr, block.size, new_ptr, new_size);

        if (new_ptr != block.ptr) {
            block.ptr = new_ptr;
        }
        block.size = new_size;
    }
}

size_t MemoryDefragmenter::get_total_fragmentation() const {
    if (blocks_.empty()) return 0;

    size_t total_free = 0;
    size_t prev_end = 0;

    for (const auto& block : blocks_) {
        size_t block_start = reinterpret_cast<size_t>(block.ptr);
        if (block_start > prev_end) {
            total_free += block_start - prev_end;
        }
        prev_end = block_start + block.size;
    }

    return total_free;
}

size_t MemoryDefragmenter::get_largest_free_block() const {
    if (blocks_.empty()) return 0;

    size_t max_free = 0;
    size_t prev_end = 0;

    for (const auto& block : blocks_) {
        size_t block_start = reinterpret_cast<size_t>(block.ptr);
        if (block_start > prev_end) {
            max_free = std::max(max_free, block_start - prev_end);
        }
        prev_end = block_start + block.size;
    }

    return max_free;
}

int MemoryDefragmenter::get_fragment_count() const {
    if (blocks_.empty()) return 0;

    int count = 0;
    size_t prev_end = 0;

    for (const auto& block : blocks_) {
        size_t block_start = reinterpret_cast<size_t>(block.ptr);
        if (block_start > prev_end) {
            count++;
        }
        prev_end = block_start + block.size;
    }

    return count;
}

void MemoryDefragmenter::set_reallocate_callback(ReallocateCallback callback) {
    reallocate_callback_ = callback;
}

MemoryOptimizationManager& MemoryOptimizationManager::instance() {
    static MemoryOptimizationManager manager;
    return manager;
}

void MemoryOptimizationManager::enable_checkpoint_compression(bool enable) {
    compression_enabled_ = enable;
}

void MemoryOptimizationManager::set_gradient_accumulation_steps(int steps) {
    gradient_accumulation_steps_ = steps;
}

void MemoryOptimizationManager::enable_defragmentation(bool enable) {
    defragmentation_enabled_ = enable;
}

void MemoryOptimizationManager::record_checkpoint_size(size_t original, size_t compressed) {
    total_original_bytes_ += original;
    total_compressed_bytes_ += compressed;
}

void MemoryOptimizationManager::record_defragmentation() {
    num_defragmentations_++;
}

MemoryOptimizationStats MemoryOptimizationManager::get_stats() const {
    MemoryOptimizationStats stats;
    stats.compressed_bytes = total_compressed_bytes_;
    stats.original_bytes = total_original_bytes_;
    stats.compression_ratio = total_original_bytes_ > 0
        ? static_cast<float>(total_original_bytes_) / total_compressed_bytes_
        : 1.0f;
    stats.gradient_accumulation_buffers = gradient_accumulation_steps_;
    stats.total_fragmentation = 0;
    stats.num_allocations = 0;
    stats.num_defragmentations = num_defragmentations_;
    return stats;
}

void MemoryOptimizationManager::reset_stats() {
    total_compressed_bytes_ = 0;
    total_original_bytes_ = 0;
    num_defragmentations_ = 0;
}

} // namespace cuda::memory_opt
