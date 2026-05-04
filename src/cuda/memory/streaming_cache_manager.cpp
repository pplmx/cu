#include "cuda/memory/streaming_cache_manager.h"
#include "cuda/device/error.h"
#include <algorithm>

namespace cuda::memory {

StreamingCacheManager::StreamingCacheManager(
    KVCacheAllocator* allocator,
    const StreamingCacheConfig& config
) : allocator_(allocator), config_(config) {}

void StreamingCacheManager::prefetch_async(
    int64_t sequence_id,
    int num_tokens_ahead,
    const stream::Stream& stream
) {
    if (!config_.enable_prefetch) {
        return;
    }

    const int prefetch_blocks = std::min(
        num_tokens_ahead / allocator_->get_block_size_tokens() + 1,
        config_.prefetch_ahead_blocks
    );

    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    PrefetchRequest request{
        .sequence_id = sequence_id,
        .num_tokens_ahead = prefetch_blocks * allocator_->get_block_size_tokens(),
        .completed = false
    };
    pending_prefetch_.push_back(request);
}

void StreamingCacheManager::evict_importance_weighted(int num_blocks) {
    if (!config_.enable_async_eviction || config_.eviction_policy != 1) {
        allocator_->evict(num_blocks);
        return;
    }

    std::vector<std::pair<int64_t, float>> sequences;
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        for (const auto& [seq_id, importance] : importance_weights_) {
            sequences.emplace_back(seq_id, importance);
        }
    }

    std::sort(sequences.begin(), sequences.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    int evicted = 0;
    for (const auto& [seq_id, _] : sequences) {
        if (evicted >= num_blocks) break;

        auto blocks = allocator_->get_blocks(seq_id);
        if (!blocks.empty()) {
            allocator_->free(seq_id);
            evicted += static_cast<int>(blocks.size());
        }
    }
}

bool StreamingCacheManager::should_evict_async() const {
    if (!config_.enable_async_eviction) {
        return false;
    }

    const int free_blocks = allocator_->get_num_free_blocks();
    const int total_blocks = 4096;
    const float free_ratio = static_cast<float>(free_blocks) / total_blocks;

    return free_ratio < 0.1f;
}

void StreamingCacheManager::update_importance(int64_t sequence_id, float importance) {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    importance_weights_[sequence_id] = importance;
}

float StreamingCacheManager::get_importance(int64_t sequence_id) const {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    auto it = importance_weights_.find(sequence_id);
    return it != importance_weights_.end() ? it->second : 1.0f;
}

void StreamingCacheManager::sync_prefetch(const stream::Stream& stream) {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);

    for (auto& request : pending_prefetch_) {
        if (!request.completed) {
            allocator_->append(request.sequence_id, request.num_tokens_ahead);
            request.completed = true;
        }
    }

    pending_prefetch_.clear();
}

}  // namespace cuda::memory
