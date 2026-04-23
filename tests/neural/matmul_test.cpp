#include <gtest/gtest.h>
#include "cuda/neural/matmul.h"

using namespace cuda::neural;

class MatmulTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(MatmulTest, GetCublasHandleReturnsValidHandle) {
    cublasHandle_t handle = get_cublas_handle();
    EXPECT_NE(handle, nullptr);
}

TEST_F(MatmulTest, MatmulOptionsHaveDefaults) {
    MatmulOptions options;
    EXPECT_EQ(options.alpha, 1.0f);
    EXPECT_EQ(options.beta, 0.0f);
    EXPECT_EQ(options.trans_a, CUBLAS_OP_N);
    EXPECT_EQ(options.trans_b, CUBLAS_OP_N);
}

TEST_F(MatmulTest, MatmulOptionsCanBeCustomized) {
    MatmulOptions options;
    options.alpha = 2.0f;
    options.beta = 1.0f;
    options.trans_a = CUBLAS_OP_T;

    EXPECT_EQ(options.alpha, 2.0f);
    EXPECT_EQ(options.beta, 1.0f);
    EXPECT_EQ(options.trans_a, CUBLAS_OP_T);
}

TEST_F(MatmulTest, MatmulOptionsWithCublasHandle) {
    cublasHandle_t handle = get_cublas_handle();
    MatmulOptions options;
    options.handle = handle;

    EXPECT_EQ(options.handle, handle);
}
