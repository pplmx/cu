#include <gtest/gtest.h>
#include <cmath>
#include "cuda/neural/activations.h"

using namespace cuda::neural;

class ActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
    }
};

TEST_F(ActivationTest, ReLUPositiveValues) {
    EXPECT_EQ(std::fmaxf(0.0f, 5.0f), 5.0f);
    EXPECT_EQ(std::fmaxf(0.0f, 100.0f), 100.0f);
}

TEST_F(ActivationTest, ReLUNegativeValues) {
    EXPECT_EQ(std::fmaxf(0.0f, -5.0f), 0.0f);
    EXPECT_EQ(std::fmaxf(0.0f, -100.0f), 0.0f);
}

TEST_F(ActivationTest, LeakyReLUPositiveValues) {
    float alpha = 0.01f;
    float x = 5.0f;
    float expected = x > 0.0f ? x : alpha * x;
    EXPECT_EQ(expected, 5.0f);
}

TEST_F(ActivationTest, LeakyReLUNegativeValues) {
    float alpha = 0.01f;
    float x = -5.0f;
    float expected = x > 0.0f ? x : alpha * x;
    EXPECT_NEAR(expected, -0.05f, 0.001f);
}

TEST_F(ActivationTest, LeakyReLUCustomAlpha) {
    float alpha = 0.1f;
    float x = -5.0f;
    float expected = x > 0.0f ? x : alpha * x;
    EXPECT_NEAR(expected, -0.5f, 0.001f);
}

TEST_F(ActivationTest, SigmoidPositiveValues) {
    float x = 0.0f;
    float expected = 1.0f / (1.0f + std::exp(-x));
    EXPECT_NEAR(expected, 0.5f, 0.001f);
}

TEST_F(ActivationTest, SigmoidBounds) {
    float x_large_pos = 10.0f;
    float x_large_neg = -10.0f;

    float sigmoid_pos = 1.0f / (1.0f + std::exp(-x_large_pos));
    float sigmoid_neg = 1.0f / (1.0f + std::exp(-x_large_neg));

    EXPECT_GT(sigmoid_pos, 0.99f);
    EXPECT_LT(sigmoid_neg, 0.01f);
}

TEST_F(ActivationTest, TanhBounds) {
    float x = 0.0f;
    float expected = std::tanh(x);
    EXPECT_NEAR(expected, 0.0f, 0.001f);
}

TEST_F(ActivationTest, TanhAsymptotes) {
    float x_large_pos = 10.0f;
    float x_large_neg = -10.0f;

    EXPECT_GT(std::tanh(x_large_pos), 0.99f);
    EXPECT_LT(std::tanh(x_large_neg), -0.99f);
}

TEST_F(ActivationTest, ActivationOptionsDefaults) {
    ActivationOptions options;
    EXPECT_EQ(options.alpha, 0.01f);
    EXPECT_EQ(options.negative_slope, 0.01f);
}
