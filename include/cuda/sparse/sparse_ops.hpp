#ifndef NOVA_CUDA_SPARSE_OPS_HPP
#define NOVA_CUDA_SPARSE_OPS_HPP

#include <nova/sparse/sparse_matrix.hpp>
#include <memory>

namespace nova {
namespace sparse {

template<typename T>
void sparse_mv(const SparseMatrixCSR<T>& matrix, const T* x, T* y);

template<typename T>
void sparse_mm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_vecs);

template<typename T>
class SparseOps {
public:
    static void spmv(const SparseMatrixCSR<T>& matrix, const T* x, T* y);
    static void spmm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_cols);
    static void spmv(const SparseMatrixELL<T>& matrix, const T* x, T* y);
    static void spmv(const SparseMatrixSELL<T>& matrix, const T* x, T* y);
};

template<typename T>
void SparseOps<T>::spmv(const SparseMatrixCSR<T>& matrix, const T* x, T* y) {
    int num_rows = matrix.num_rows();

    for (int i = 0; i < num_rows; ++i) {
        T sum = T{0};
        for (int idx = matrix.row_offsets()[i]; idx < matrix.row_offsets()[i + 1]; ++idx) {
            int col = matrix.col_indices()[idx];
            sum += matrix.values()[idx] * x[col];
        }
        y[i] = sum;
    }
}

template<typename T>
void SparseOps<T>::spmm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_cols) {
    int num_rows = matrix.num_rows();

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            T sum = T{0};
            for (int idx = matrix.row_offsets()[i]; idx < matrix.row_offsets()[i + 1]; ++idx) {
                int col = matrix.col_indices()[idx];
                sum += matrix.values()[idx] * B[col * num_cols + j];
            }
            C[i * num_cols + j] = sum;
        }
    }
}

template<typename T>
void sparse_mv(const SparseMatrixCSR<T>& matrix, const T* x, T* y) {
    SparseOps<T>::spmv(matrix, x, y);
}

template<typename T>
void sparse_mm(const SparseMatrixCSR<T>& matrix, const T* B, T* C, int num_vecs) {
    SparseOps<T>::spmm(matrix, B, C, num_vecs);
}

template<typename T>
void SparseOps<T>::spmv(const SparseMatrixELL<T>& matrix, const T* x, T* y) {
    int num_rows = matrix.num_rows();
    int max_nnz = matrix.max_nnz_per_row();

    for (int i = 0; i < num_rows; ++i) {
        T sum = T{0};
        int base = i * max_nnz;
        for (int j = 0; j < max_nnz; ++j) {
            int col = matrix.col_indices()[base + j];
            if (col >= 0) {
                sum += matrix.values()[base + j] * x[col];
            }
        }
        y[i] = sum;
    }
}

template<typename T>
void SparseOps<T>::spmv(const SparseMatrixSELL<T>& matrix, const T* x, T* y) {
    int num_rows = matrix.num_rows();
    int slice_height = matrix.slice_height();
    int num_slices = (num_rows + slice_height - 1) / slice_height;

    for (int s = 0; s < num_slices; ++s) {
        int slice_start_row = s * slice_height;
        int slice_end_row = std::min(slice_start_row + slice_height, num_rows);

        int slice_base = matrix.slice_ptr()[s];
        int slice_next_base = matrix.slice_ptr()[s + 1];
        int slice_nnz = (slice_next_base - slice_base) / slice_height;

        for (int local_i = 0; local_i < slice_end_row - slice_start_row; ++local_i) {
            int global_i = slice_start_row + local_i;
            T sum = T{0};
            int base = slice_base + local_i * slice_nnz;

            for (int j = 0; j < slice_nnz; ++j) {
                int col = matrix.col_indices()[base + j];
                if (col >= 0) {
                    sum += matrix.values()[base + j] * x[col];
                }
            }
            y[global_i] = sum;
        }
    }
}

template<typename T>
void sparse_mv(const SparseMatrixELL<T>& matrix, const T* x, T* y) {
    SparseOps<T>::spmv(matrix, x, y);
}

template<typename T>
void sparse_mv(const SparseMatrixSELL<T>& matrix, const T* x, T* y) {
    SparseOps<T>::spmv(matrix, x, y);
}

} // namespace sparse
} // namespace nova

#endif // NOVA_CUDA_SPARSE_OPS_HPP
