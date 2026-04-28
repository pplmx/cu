#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "cuda/device/error.h"

namespace cuda::production {

enum class StreamPriority {
    Low = -1,
    Normal = 0,
    High = 1,
};

class PriorityStream {
public:
    PriorityStream() = default;
    PriorityStream(cudaStream_t stream, StreamPriority priority)
        : stream_(stream), priority_(priority) {}

    [[nodiscard]] cudaStream_t get() const { return stream_; }
    [[nodiscard]] StreamPriority priority() const { return priority_; }
    [[nodiscard]] explicit operator bool() const { return stream_ != nullptr; }

private:
    cudaStream_t stream_{nullptr};
    StreamPriority priority_{StreamPriority::Normal};
};

class PriorityStreamPool {
public:
    static constexpr int MIN_PRIORITY = -2;
    static constexpr int MAX_PRIORITY = 0;

    PriorityStreamPool() = default;

    explicit PriorityStreamPool(size_t pool_size);

    ~PriorityStreamPool() {
        cleanup();
    }

    PriorityStreamPool(const PriorityStreamPool&) = delete;
    PriorityStreamPool& operator=(const PriorityStreamPool&) = delete;

    [[nodiscard]] PriorityStream acquire(StreamPriority priority);
    void release(PriorityStream stream);

    [[nodiscard]] size_t available_count(StreamPriority priority) const;
    [[nodiscard]] size_t total_available() const;

    void cleanup();

private:
    std::vector<PriorityStream> create_streams(size_t count, StreamPriority priority);

    std::vector<PriorityStream> low_priority_pool_;
    std::vector<PriorityStream> normal_priority_pool_;
    std::vector<PriorityStream> high_priority_pool_;

    mutable std::mutex mutex_;
    size_t pool_size_ = 4;
};

}  // namespace cuda::production
