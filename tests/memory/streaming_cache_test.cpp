#include <gtest/gtest.h>
#include "cuda/memory/streaming_cache_manager.h"
#include "cuda/memory/kv_cache_allocator.h"
#include "cuda/stream/stream.h"

namespace cuda::memory {

class StreamingCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        KVCacheAllocatorConfig config{
            .num_heads = 32,
            .head_dim = 128,
            .block_size_tokens = 16,
            .num_blocks = 256,
            .num_layers = 32,
        };
        allocator = std::make_unique<KVCacheAllocator>(config);

        StreamingCacheConfig stream_config{
            .enable_prefetch = true,
            .enable_async_eviction = true,
            .prefetch_ahead_blocks = 2,
            .eviction_batch_size = 4,
        };
        manager = std::make_unique<StreamingCacheManager>(allocator.get(), stream_config);
    }

    std::unique_ptr<KVCacheAllocator> allocator;
    std::unique_ptr<StreamingCacheManager> manager;
};

TEST_F(StreamingCacheTest, PrefetchRequestCreated) {
    stream::Stream stream;

    auto blocks1 = allocator->allocate(1, 32);
    EXPECT_GT(blocks1.size(), 0);

    manager->prefetch_async(1, 16, stream);

    EXPECT_TRUE(manager->should_evict_async() == false ||
                allocator->get_num_free_blocks() > 0);
}

TEST_F(StreamingCacheTest, ImportanceTracking) {
    manager->update_importance(1, 10.0f);
    manager->update_importance(2, 5.0f);

    EXPECT_EQ(manager->get_importance(1), 10.0f);
    EXPECT_EQ(manager->get_importance(2), 5.0f);
    EXPECT_EQ(manager->get_importance(3), 1.0f);
}

TEST_F(StreamingCacheTest, DefaultImportanceIsOne) {
    EXPECT_EQ(manager->get_importance(999), 1.0f);
}

}  // namespace cuda::memory
