#pragma once

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <vector>

#include "cuda/memory/buffer.h"

namespace nova::sparse {

template<typename T>
class SparseMatrix {
public:
    SparseMatrix() = default;

    SparseMatrix(int num_rows, int num_cols, int nnz)
        : values_(static_cast<size_t>(nnz))
        , row_offsets_(static_cast<size_t>(num_rows + 1))
        , col_indices_(static_cast<size_t>(nnz))
        , num_rows_(num_rows)
        , num_cols_(num_cols) {}

    static std::optional<SparseMatrix> FromDense(const T* dense, int rows, int cols,
                                                  float sparsity_threshold = 0.0f);

    static SparseMatrix FromHostData(std::vector<T> values,
                                      std::vector<int> row_offsets,
                                      std::vector<int> col_indices,
                                      int num_rows, int num_cols);

    static SparseMatrix FromEdgeList(const std::vector<std::pair<int, int>>& edges,
                                      int num_vertices,
                                      const std::vector<T>* weights = nullptr);

    int rows() const { return num_rows_; }
    int cols() const { return num_cols_; }
    int nnz() const { return static_cast<int>(values_.size()); }

    T* values() { return values_.data(); }
    int* row_offsets() { return row_offsets_.data(); }
    int* col_indices() { return col_indices_.data(); }

    const T* values() const { return values_.data(); }
    const int* row_offsets() const { return row_offsets_.data(); }
    const int* col_indices() const { return col_indices_.data(); }

    cuda::memory::Buffer<T>& values_buffer() { return values_; }
    cuda::memory::Buffer<int>& row_offsets_buffer() { return row_offsets_; }
    cuda::memory::Buffer<int>& col_indices_buffer() { return col_indices_; }

    const cuda::memory::Buffer<T>& values_buffer() const { return values_; }
    const cuda::memory::Buffer<int>& row_offsets_buffer() const { return row_offsets_; }
    const cuda::memory::Buffer<int>& col_indices_buffer() const { return col_indices_; }

    void copy_to_host(std::vector<T>& out_values,
                      std::vector<int>& out_row_offsets,
                      std::vector<int>& out_col_indices) const;

private:
    cuda::memory::Buffer<T> values_;
    cuda::memory::Buffer<int> row_offsets_;
    cuda::memory::Buffer<int> col_indices_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};

template<typename T>
std::optional<SparseMatrix<T>> SparseMatrix<T>::FromDense(const T* dense,
                                                          int rows, int cols,
                                                          float threshold) {
    std::vector<T> values;
    std::vector<int> row_offsets(1, 0);
    std::vector<int> col_indices;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T val = dense[i * cols + j];
            if (val != T{0} && std::abs(val) > threshold) {
                values.push_back(val);
                col_indices.push_back(j);
            }
        }
        row_offsets.push_back(static_cast<int>(values.size()));
    }

    if (values.empty()) {
        return std::nullopt;
    }

    SparseMatrix<T> result(rows, cols, static_cast<int>(values.size()));
    result.values_buffer().copy_from(values.data(), static_cast<size_t>(values.size()));
    result.row_offsets_buffer().copy_from(row_offsets.data(), static_cast<size_t>(row_offsets.size()));
    result.col_indices_buffer().copy_from(col_indices.data(), static_cast<size_t>(col_indices.size()));

    return result;
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>::FromHostData(std::vector<T> values,
                                               std::vector<int> row_offsets,
                                               std::vector<int> col_indices,
                                               int num_rows, int num_cols) {
    SparseMatrix<T> result(num_rows, num_cols, static_cast<int>(values.size()));
    result.values_buffer().copy_from(values.data(), static_cast<size_t>(values.size()));
    result.row_offsets_buffer().copy_from(row_offsets.data(), static_cast<size_t>(row_offsets.size()));
    result.col_indices_buffer().copy_from(col_indices.data(), static_cast<size_t>(col_indices.size()));
    return result;
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>::FromEdgeList(const std::vector<std::pair<int, int>>& edges,
                                              int num_vertices,
                                              const std::vector<T>* weights) {
    std::vector<int> row_offsets(num_vertices + 1, 0);
    std::vector<int> col_indices;
    std::vector<T> values;

    col_indices.reserve(edges.size());
    values.reserve(edges.size());

    for (const auto& edge : edges) {
        col_indices.push_back(edge.second);
        values.push_back(weights ? (*weights)[&edge - &edges[0]] : T{1});
    }

    for (int i = 0; i <= num_vertices; ++i) {
        row_offsets[i] = i == 0 ? 0 : static_cast<int>(edges.size());
    }

    return FromHostData(values, row_offsets, col_indices, num_vertices, num_vertices);
}

template<typename T>
void SparseMatrix<T>::copy_to_host(std::vector<T>& out_values,
                                    std::vector<int>& out_row_offsets,
                                    std::vector<int>& out_col_indices) const {
    out_values.resize(static_cast<size_t>(nnz()));
    out_row_offsets.resize(static_cast<size_t>(num_rows_ + 1));
    out_col_indices.resize(static_cast<size_t>(nnz()));

    values_.copy_to(out_values.data(), static_cast<size_t>(nnz()));
    row_offsets_.copy_to(out_row_offsets.data(), static_cast<size_t>(num_rows_ + 1));
    col_indices_.copy_to(out_col_indices.data(), static_cast<size_t>(nnz()));
}

template<typename T>
void spmv(const SparseMatrix<T>& A, const T* x, T* y);

template<typename T>
void spmv_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream);

template<typename T>
void spmv_transpose(const SparseMatrix<T>& A, const T* x, T* y);

template<typename T>
void spmv_transpose_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream);

template<typename T>
class SparseMatrixCSR;

template<typename T>
SparseMatrix<T> ToSparseMatrix(const SparseMatrixCSR<T>& csr) {
    if (csr.nnz() == 0) {
        return SparseMatrix<T>(csr.num_rows(), csr.num_cols(), 0);
    }

    std::vector<T> values(csr.nnz());
    std::vector<int> row_offsets(csr.num_rows() + 1);
    std::vector<int> col_indices(csr.nnz());

    std::copy(csr.values(), csr.values() + csr.nnz(), values.begin());
    std::copy(csr.row_offsets(), csr.row_offsets() + csr.num_rows() + 1, row_offsets.begin());
    std::copy(csr.col_indices(), csr.col_indices() + csr.nnz(), col_indices.begin());

    return SparseMatrix<T>::FromHostData(values, row_offsets, col_indices,
                                          csr.num_rows(), csr.num_cols());
}

}  // namespace nova::sparse
