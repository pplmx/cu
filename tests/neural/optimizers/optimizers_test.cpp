#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda/neural/optimizers/optimizers.h>

namespace cuda::neural::optimizers::test {

class OptimizersTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetDevice(&device_);
    }

    int device_ = 0;
};

TEST_F(OptimizersTest, AdamWOptimizerConstruction) {
    OptimizerConfig config;
    config.learning_rate = 0.001f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.weight_decay = 0.01f;

    AdamWOptimizer optimizer(config);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.001f);
    EXPECT_EQ(optimizer.get_weight_decay(), 0.01f);
}

TEST_F(OptimizersTest, AdamWOptimizerSetters) {
    OptimizerConfig config;
    AdamWOptimizer optimizer(config);

    optimizer.set_learning_rate(0.01f);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.01f);

    optimizer.set_weight_decay(0.001f);
    EXPECT_EQ(optimizer.get_weight_decay(), 0.001f);
}

TEST_F(OptimizersTest, AdamWOptimizerZeroMomentum) {
    OptimizerConfig config;
    AdamWOptimizer optimizer(config);
    optimizer.zero_momentum();
}

TEST_F(OptimizersTest, LAMBOptimizerConstruction) {
    LAMBConfig config;
    config.learning_rate = 0.001f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.use_layer_adaptation = true;

    LAMBOptimizer optimizer(config);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.001f);
}

TEST_F(OptimizersTest, LAMBOptimizerSetters) {
    LAMBConfig config;
    LAMBOptimizer optimizer(config);

    optimizer.set_learning_rate(0.005f);
    EXPECT_EQ(optimizer.get_learning_rate(), 0.005f);
}

TEST_F(OptimizersTest, LAMBOptimizerZeroMomentum) {
    LAMBConfig config;
    LAMBOptimizer optimizer(config);
    optimizer.zero_momentum();
}

TEST_F(OptimizersTest, LAMBConfigDefaults) {
    LAMBConfig config;
    EXPECT_EQ(config.beta1, 0.9f);
    EXPECT_EQ(config.beta2, 0.999f);
    EXPECT_EQ(config.epsilon, 1e-6f);
    EXPECT_TRUE(config.use_layer_adaptation);
}

TEST_F(OptimizersTest, GradientClipperConstruction) {
    GradientClipConfig config;
    config.max_norm = 1.0f;
    config.norm_type = GradientClipConfig::NormType::L2;

    GradientClipper clipper(config);
    EXPECT_EQ(clipper.get_max_norm(), 1.0f);
}

TEST_F(OptimizersTest, GradientClipperSetters) {
    GradientClipConfig config;
    GradientClipper clipper(config);

    clipper.set_max_norm(5.0f);
    EXPECT_EQ(clipper.get_max_norm(), 5.0f);
}

TEST_F(OptimizersTest, GradientClipConfigEnums) {
    GradientClipConfig config;
    config.norm_type = GradientClipConfig::NormType::L2;
    EXPECT_EQ(config.norm_type, GradientClipConfig::NormType::L2);

    config.norm_type = GradientClipConfig::NormType::Inf;
    EXPECT_EQ(config.norm_type, GradientClipConfig::NormType::Inf);
}

TEST_F(OptimizersTest, OptimizerConfigDefaults) {
    OptimizerConfig config;
    EXPECT_EQ(config.learning_rate, 0.001f);
    EXPECT_EQ(config.beta1, 0.9f);
    EXPECT_EQ(config.beta2, 0.999f);
    EXPECT_EQ(config.epsilon, 1e-8f);
    EXPECT_EQ(config.weight_decay, 0.01f);
    EXPECT_TRUE(config.fused);
}

TEST_F(OptimizersTest, LAMBOptimizerLayerAdaptation) {
    LAMBConfig config;
    config.use_layer_adaptation = true;
    EXPECT_TRUE(config.use_layer_adaptation);

    LAMBOptimizer optimizer(config);
}

TEST_F(OptimizersTest, GradientClipperL2Config) {
    GradientClipConfig config;
    config.max_norm = 2.0f;
    config.norm_type = GradientClipConfig::NormType::L2;

    GradientClipper clipper(config);
    EXPECT_EQ(clipper.get_max_norm(), 2.0f);
}

}  // namespace cuda::neural::optimizers::test
