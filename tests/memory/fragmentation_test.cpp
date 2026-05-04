#include <gtest/gtest.h>
#include "cuda/memory/kv_cache_allocator.h"

namespace cuda::memory {

class FragmentationTest : public ::testing::Test {
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
    }

    std::unique_ptr<KVCacheAllocator> allocator;
};

TEST_F(FragmentationTest, AnalyzeFragmentation) {
    auto report = allocator->analyze_fragmentation();

    EXPECT_GE(report.num_holes, 0);
    EXPECT_GE(report.ratio, 0.0f);
}

TEST_F(FragmentationTest, NeedsCompactionThreshold) {
    EXPECT_FALSE(allocator->needs_compaction(30.0f));

    EXPECT_TRUE(allocator->needs_compaction(0.0f));
}

TEST_F(FragmentationTest, FragmentationAfterAllocations) {
    for (int i = 0; i < 10; ++i) {
        allocator->allocate(i, 32);
    }

    auto report_before = allocator->analyze_fragmentation();
    EXPECT_LT(report_before.num_holes, 256);

    for (int i = 0; i < 5; ++i) {
        allocator->free(i);
    }

    auto report_after = allocator->analyze_fragmentation();
    EXPECT_GT(report_after.num_holes, report_before.num_holes);
}

TEST_F(FragmentationTest, CompactReducesHoles) {
    for (int i = 0; i < 50; ++i) {
        allocator->allocate(i, 16);
    }

    for (int i = 0; i < 25; ++i) {
        allocator->free(i);
    }

    auto report_before = allocator->analyze_fragmentation();

    allocator->compact();

    auto report_after = allocator->analyze_fragmentation();

    EXPECT_LE(report_after.num_holes, report_before.num_holes);
}

}  // namespace cuda::memory
