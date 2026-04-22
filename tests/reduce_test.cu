#include <gtest/gtest.h>
#include "reduce.h"
#include "cuda_utils.h"
#include <numeric>

class ReduceTest : public ::testing::Test {
protected:
    size_t size_ = 1024;
    std::vector<int> h_input_;
    int *d_input_;

    void SetUp() override {
        h_input_.resize(size_);
        CUDA_CHECK(cudaMalloc(&d_input_, size_ * sizeof(int)));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(d_input_));
    }
};

TEST_F(ReduceTest, SumBasic) {
    for (int i = 1; i <= static_cast<int>(size_); ++i) h_input_[i-1] = i;
    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));

    int result = reduceSum(d_input_, size_);
    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;

    EXPECT_EQ(result, expected);
}

TEST_F(ReduceTest, SumOptimized) {
    for (int i = 1; i <= static_cast<int>(size_); ++i) h_input_[i-1] = i;
    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));

    int result = reduceSumOptimized(d_input_, size_);
    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;

    EXPECT_EQ(result, expected);
}

TEST_F(ReduceTest, SumConsistency) {
    for (int i = 1; i <= static_cast<int>(size_); ++i) h_input_[i-1] = i;
    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));

    int basic = reduceSum(d_input_, size_);
    int optimized = reduceSumOptimized(d_input_, size_);

    EXPECT_EQ(basic, optimized);
}

TEST_F(ReduceTest, MaxTest) {
    h_input_.assign(size_, 0);
    h_input_[500] = 999;
    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));

    int result = reduceMax(d_input_, size_);
    EXPECT_EQ(result, 999);
}

TEST_F(ReduceTest, MinTest) {
    for (int i = 0; i < static_cast<int>(size_); ++i) h_input_[i] = i + 100;
    CUDA_CHECK(cudaMemcpy(d_input_, h_input_.data(), size_ * sizeof(int), cudaMemcpyHostToDevice));

    int result = reduceMin(d_input_, size_);
    EXPECT_EQ(result, 100);
}
