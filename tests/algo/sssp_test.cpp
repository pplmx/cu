#include <gtest/gtest.h>

#include "cuda/algo/sssp.h"
#include "cuda/graph/csr_graph.h"

namespace cuda::algo::sssp::test {

class SSSPTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SSSPTest, ConfigSetters) {
    SSSPConfig config;
    config.default_delta = 2.0f;
    config.max_iterations = 500;

    cuda::algo::sssp::set_config(config);

    auto retrieved = cuda::algo::sssp::get_config();
    EXPECT_EQ(retrieved.default_delta, 2.0f);
    EXPECT_EQ(retrieved.max_iterations, 500);
}

TEST_F(SSSPTest, INFConstant) {
    EXPECT_TRUE(cuda::algo::sssp::INF > 1e20f);
    EXPECT_FALSE(cuda::algo::sssp::INF < cuda::algo::sssp::INF);
}

TEST_F(SSSPTest, BellmanFordSimple) {
    graph::CSRGraph graph;

    constexpr int num_vertices = 4;
    constexpr int num_edges = 5;

    std::vector<float> values = {1.0f, 4.0f, 2.0f, 5.0f, 1.0f};
    std::vector<int> row_offsets = {0, 2, 4, 5, 5};
    std::vector<int> col_indices = {1, 2, 0, 3, 2};

    cuda::memory::Buffer<float> d_values(values.size());
    cuda::memory::Buffer<int> d_row_offsets(row_offsets.size());
    cuda::memory::Buffer<int> d_col_indices(col_indices.size());

    cudaMemcpy(d_values.data(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets.data(), row_offsets.data(), row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices.data(), col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    graph = graph::CSRGraph(num_vertices, num_edges,
                            d_row_offsets.data(), d_col_indices.data(),
                            d_values.data());

    memory::Buffer<float> distances(num_vertices);

    cuda::algo::sssp::bellman_ford(graph, 0, distances.data());

    std::vector<float> result(num_vertices);
    cudaMemcpy(result.data(), distances.data(), num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[0], 0.0f);
    EXPECT_EQ(result[1], 1.0f);
    EXPECT_LE(result[2], 3.0f);
}

TEST_F(SSSPTest, DeltaSteppingSimple) {
    graph::CSRGraph graph;

    constexpr int num_vertices = 3;
    constexpr int num_edges = 3;

    std::vector<float> values = {1.0f, 1.0f, 1.0f};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {1, 2, 0};

    cuda::memory::Buffer<float> d_values(values.size());
    cuda::memory::Buffer<int> d_row_offsets(row_offsets.size());
    cuda::memory::Buffer<int> d_col_indices(col_indices.size());

    cudaMemcpy(d_values.data(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets.data(), row_offsets.data(), row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices.data(), col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    graph = graph::CSRGraph(num_vertices, num_edges,
                            d_row_offsets.data(), d_col_indices.data(),
                            d_values.data());

    memory::Buffer<float> distances(num_vertices);

    cuda::algo::sssp::delta_stepping(graph, 0, distances.data(), 1.0f);

    std::vector<float> result(num_vertices);
    cudaMemcpy(result.data(), distances.data(), num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[0], 0.0f);
    EXPECT_EQ(result[1], 1.0f);
    EXPECT_EQ(result[2], 2.0f);
}

TEST_F(SSSPTest, ComputeDistances) {
    graph::CSRGraph graph;

    constexpr int num_vertices = 2;
    constexpr int num_edges = 1;

    std::vector<float> values = {5.0f};
    std::vector<int> row_offsets = {0, 1, 1};
    std::vector<int> col_indices = {1};

    cuda::memory::Buffer<float> d_values(values.size());
    cuda::memory::Buffer<int> d_row_offsets(row_offsets.size());
    cuda::memory::Buffer<int> d_col_indices(col_indices.size());

    cudaMemcpy(d_values.data(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offsets.data(), row_offsets.data(), row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices.data(), col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    graph = graph::CSRGraph(num_vertices, num_edges,
                            d_row_offsets.data(), d_col_indices.data(),
                            d_values.data());

    auto distances = cuda::algo::sssp::compute_distances<float>(graph, 0, 1.0f);

    std::vector<float> result(num_vertices);
    cudaMemcpy(result.data(), distances.data(), num_vertices * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(result[0], 0.0f);
    EXPECT_EQ(result[1], 5.0f);
}

}  // namespace cuda::algo::sssp::test
