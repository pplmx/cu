#include <gtest/gtest.h>
#include <cuda/sparse/matrix.hpp>

namespace nova::sparse::test {

template<typename T>
bool approx_equal(T a, T b, T tol = T{1e-5}) {
    return std::abs(a - b) < tol;
}

TEST(SparseMatrixTest, DefaultConstruction) {
    SparseMatrix<float> matrix;
    EXPECT_EQ(matrix.rows(), 0);
    EXPECT_EQ(matrix.cols(), 0);
    EXPECT_EQ(matrix.nnz(), 0);
}

TEST(SparseMatrixTest, ConstructionWithDimensions) {
    SparseMatrix<float> matrix(3, 3, 5);
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nnz(), 5);
    EXPECT_NE(matrix.values(), nullptr);
    EXPECT_NE(matrix.row_offsets(), nullptr);
    EXPECT_NE(matrix.col_indices(), nullptr);
}

TEST(SparseMatrixTest, FromDenseTrivial) {
    std::vector<float> dense = {4.0f, 1.0f, 1.0f, 3.0f};
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 2, 2);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 2);
    EXPECT_EQ(matrix->cols(), 2);
    EXPECT_EQ(matrix->nnz(), 4);
}

TEST(SparseMatrixTest, FromDenseSparse) {
    std::vector<float> dense = {
        4.0f, 0.0f, 1.0f,
        0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 2.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 3);
    EXPECT_EQ(matrix->cols(), 3);
    EXPECT_EQ(matrix->nnz(), 4);
}

TEST(SparseMatrixTest, FromDenseWithThreshold) {
    std::vector<float> dense = {
        4.0f, 0.001f, 1.0f,
        0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 2.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.1f);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->nnz(), 3);
}

TEST(SparseMatrixTest, FromDenseAllZeros) {
    std::vector<float> dense(9, 0.0f);
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3);

    EXPECT_FALSE(matrix.has_value());
}

TEST(SparseMatrixTest, FromHostData) {
    std::vector<float> values = {4.0f, 1.0f, 1.0f, 3.0f};
    std::vector<int> row_offsets = {0, 2, 4};
    std::vector<int> col_indices = {0, 1, 0, 1};

    auto matrix = SparseMatrix<float>::FromHostData(values, row_offsets, col_indices, 2, 2);

    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_EQ(matrix.nnz(), 4);
}

TEST(SparseMatrixTest, FromEdgeList) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 3);

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nnz(), 3);
}

TEST(SparseMatrixTest, FromEdgeListWithWeights) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 0}};
    std::vector<float> weights = {0.5f, 1.5f, 2.0f};
    auto matrix = SparseMatrix<float>::FromEdgeList(edges, 3, &weights);

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.nnz(), 3);

    std::vector<float> h_values;
    std::vector<int> h_row_offsets, h_col_indices;
    matrix.copy_to_host(h_values, h_row_offsets, h_col_indices);

    EXPECT_FLOAT_EQ(h_values[0], 0.5f);
    EXPECT_FLOAT_EQ(h_values[1], 1.5f);
    EXPECT_FLOAT_EQ(h_values[2], 2.0f);
}

TEST(SparseMatrixTest, CopyToHost) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> row_offsets = {0, 1, 2, 3, 4};
    std::vector<int> col_indices = {0, 1, 2, 3};

    auto matrix = SparseMatrix<float>::FromHostData(values, row_offsets, col_indices, 4, 4);

    std::vector<float> out_values;
    std::vector<int> out_row_offsets, out_col_indices;
    matrix.copy_to_host(out_values, out_row_offsets, out_col_indices);

    EXPECT_EQ(out_values, values);
    EXPECT_EQ(out_row_offsets, row_offsets);
    EXPECT_EQ(out_col_indices, col_indices);
}

TEST(SparseMatrixTest, DoublePrecision) {
    std::vector<double> dense = {4.0, 1.0, 1.0, 3.0};
    auto matrix = SparseMatrix<double>::FromDense(dense.data(), 2, 2);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 2);
    EXPECT_EQ(matrix->cols(), 2);
    EXPECT_EQ(matrix->nnz(), 4);
}

TEST(SparseMatrixTest, SPDMatrix) {
    std::vector<float> dense = {
        4.0f, 1.0f, 0.0f,
        1.0f, 4.0f, 1.0f,
        0.0f, 1.0f, 4.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3);

    ASSERT_TRUE(matrix.has_value());
    EXPECT_EQ(matrix->rows(), 3);
    EXPECT_EQ(matrix->cols(), 3);
    EXPECT_EQ(matrix->nnz(), 7);
}

}  // namespace nova::sparse::test
