#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <functional>

namespace cuda::memory_opt {

struct CompressionConfig {
    bool enable_compression = true;
    int compression_level = 3;
    size_t min_size_for_compression = 1024;
};

class CheckpointCompressor {
public:
    static CheckpointCompressor& instance();

    void set_config(const CompressionConfig& config);
    CompressionConfig get_config() const;

    size_t compress(
        const void* input,
        size_t input_size,
        void* output,
        size_t output_capacity
    );

    size_t decompress(
        const void* input,
        size_t input_size,
        void* output,
        size_t output_capacity
    );

    float get_compression_ratio() const { return compression_ratio_; }

private:
    CheckpointCompressor() = default;

    CompressionConfig config_;
    float compression_ratio_ = 1.0f;
};

class GradientAccumulator {
public:
    GradientAccumulator(int max_accumulation_steps);
    ~GradientAccumulator();

    void add_gradient(int step, const float* gradient, size_t size);
    void get_accumulated_gradient(float* output);
    void reset();
    bool is_ready_to_apply() const;

    int current_step() const { return current_step_; }
    int max_steps() const { return max_accumulation_steps_; }

private:
    int max_accumulation_steps_;
    int current_step_ = 0;
    std::vector<float> accumulated_gradients_;
    bool has_gradients_ = false;
};

struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    int device_id;
};

class MemoryDefragmenter {
public:
    MemoryDefragmenter(int device_id);

    void register_allocation(void* ptr, size_t size);
    void unregister_allocation(void* ptr);
    void defragment();

    size_t get_total_fragmentation() const;
    size_t get_largest_free_block() const;
    int get_fragment_count() const;

    using ReallocateCallback = std::function<void(void* old_ptr, size_t old_size, void*& new_ptr, size_t& new_size)>;
    void set_reallocate_callback(ReallocateCallback callback);

private:
    int device_id_;
    std::vector<MemoryBlock> blocks_;
    ReallocateCallback reallocate_callback_;
};

struct MemoryOptimizationStats {
    size_t compressed_bytes;
    size_t original_bytes;
    float compression_ratio;
    size_t gradient_accumulation_buffers;
    size_t total_fragmentation;
    int num_allocations;
    int num_defragmentations;
};

class MemoryOptimizationManager {
public:
    static MemoryOptimizationManager& instance();

    void enable_checkpoint_compression(bool enable);
    void set_gradient_accumulation_steps(int steps);
    void enable_defragmentation(bool enable);

    void record_checkpoint_size(size_t original, size_t compressed);
    void record_defragmentation();

    MemoryOptimizationStats get_stats() const;
    void reset_stats();

private:
    MemoryOptimizationManager() = default;

    bool compression_enabled_ = true;
    bool defragmentation_enabled_ = true;
    int gradient_accumulation_steps_ = 1;

    size_t total_compressed_bytes_ = 0;
    size_t total_original_bytes_ = 0;
    int num_defragmentations_ = 0;
};

} // namespace cuda::memory_opt
