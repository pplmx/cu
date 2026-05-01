#include "cuda/sparse/preconditioner.hpp"
#include "cuda/sparse/cusparse_context.hpp"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace nova::sparse {

template<typename T>
class ILUPreconditionerImpl {
public:
    void setup(const SparseMatrix<T>& A);

    void apply(const T* in, T* out);

    void apply(const memory::Buffer<T>& in, memory::Buffer<T>& out);

    double fill_in_ratio() const { return fill_in_ratio_; }

private:
    void compute_fill_in_ratio(const SparseMatrix<T>& A);

    memory::Buffer<T> buffer_;
    memory::Buffer<int> pivot_indices_;
    int n_ = 0;
    double fill_in_ratio_ = 0.0;
};

template<typename T>
void ILUPreconditionerImpl<T>::compute_fill_in_ratio(const SparseMatrix<T>& A) {
    if (buffer_.size() == 0) {
        fill_in_ratio_ = 1.0;
        return;
    }

    int nnz_ilu = static_cast<int>(buffer_.size());
    int nnz_original = A.nnz();

    if (nnz_original > 0) {
        fill_in_ratio_ = static_cast<double>(nnz_ilu) / nnz_original;
    } else {
        fill_in_ratio_ = 0.0;
    }
}

template<>
void ILUPreconditionerImpl<double>::setup(const SparseMatrix<double>& A) {
    n_ = A.rows();
    const int nnz = A.nnz();

    buffer_.resize(nnz);
    pivot_indices_.resize(n_);

    std::vector<double> h_values(nnz);
    std::vector<int> h_row_offsets(n_ + 1);
    std::vector<int> h_col_indices(nnz);

    A.copy_to_host(h_values, h_row_offsets, h_col_indices);

    auto& ctx = detail::CusparseContext::get();

    cusparseMatDescr_t descr;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

    buffer_.copy_from(h_values.data(), nnz);
    memory::Buffer<int> d_row_offsets(n_ + 1);
    memory::Buffer<int> d_col_indices(nnz);
    d_row_offsets.copy_from(h_row_offsets.data(), n_ + 1);
    d_col_indices.copy_from(h_col_indices.data(), nnz);

    int buffer_size = 0;
    CUSPARSE_CHECK(cusparseDcsrilu0_bufferSize(
        ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        n_, descr, buffer_.data(),
        d_row_offsets.data(), d_col_indices.data(),
        &buffer_size));

    memory::Buffer<char> workspace(buffer_size);

    CUSPARSE_CHECK(cusparseDcsrilu0_analysis(
        ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        n_, descr, buffer_.data(),
        d_row_offsets.data(), d_col_indices.data(),
        pivot_indices_.data(),
        CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        workspace.data()));

    cusparseStatus_t status = cusparseDcsrilu0(
        ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        n_, descr, buffer_.data(),
        d_row_offsets.data(), d_col_indices.data(),
        pivot_indices_.data(),
        CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        workspace.data());

    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        throw PreconditionerError("ILUPreconditioner: zero pivot detected during factorization");
    }

    CUSPARSE_CHECK(status);

    compute_fill_in_ratio(A);

    cusparseDestroyMatDescr(descr);
}

template<>
void ILUPreconditionerImpl<float>::setup(const SparseMatrix<float>& A) {
    n_ = A.rows();
    const int nnz = A.nnz();

    buffer_.resize(nnz);
    pivot_indices_.resize(n_);

    std::vector<float> h_values(nnz);
    std::vector<int> h_row_offsets(n_ + 1);
    std::vector<int> h_col_indices(nnz);

    A.copy_to_host(h_values, h_row_offsets, h_col_indices);

    auto& ctx = detail::CusparseContext::get();

    cusparseMatDescr_t descr;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

    buffer_.copy_from(h_values.data(), nnz);
    memory::Buffer<int> d_row_offsets(n_ + 1);
    memory::Buffer<int> d_col_indices(nnz);
    d_row_offsets.copy_from(h_row_offsets.data(), n_ + 1);
    d_col_indices.copy_from(h_col_indices.data(), nnz);

    int buffer_size = 0;
    CUSPARSE_CHECK(cusparseScsrilu0_bufferSize(
        ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        n_, descr, buffer_.data(),
        d_row_offsets.data(), d_col_indices.data(),
        &buffer_size));

    memory::Buffer<char> workspace(buffer_size);

    CUSPARSE_CHECK(cusparseScsrilu0_analysis(
        ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        n_, descr, buffer_.data(),
        d_row_offsets.data(), d_col_indices.data(),
        pivot_indices_.data(),
        CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        workspace.data()));

    cusparseStatus_t status = cusparseScsrilu0(
        ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        n_, descr, buffer_.data(),
        d_row_offsets.data(), d_col_indices.data(),
        pivot_indices_.data(),
        CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        workspace.data());

    if (status == CUSPARSE_STATUS_ZERO_PIVOT) {
        throw PreconditionerError("ILUPreconditioner: zero pivot detected during factorization");
    }

    CUSPARSE_CHECK(status);

    compute_fill_in_ratio(A);

    cusparseDestroyMatDescr(descr);
}

template<typename T>
void ILUPreconditionerImpl<T>::apply(const T* in, T* out) {
    std::vector<T> h_in(n_);
    std::vector<T> h_out(n_);
    std::copy(in, in + n_, h_in.begin());

    apply(memory::Buffer<T>(), memory::Buffer<T>());

    for (int i = 0; i < n_; ++i) {
        out[i] = h_in[i];
    }
}

template<typename T>
void ILUPreconditionerImpl<T>::apply(const memory::Buffer<T>& in, memory::Buffer<T>& out) {
    out.resize(n_);
    std::vector<T> h_in(n_);
    std::vector<T> h_out(n_);
    in.copy_to(h_in.data(), n_);

    auto& ctx = detail::CusparseContext::get();

    cusparseMatDescr_t descr;
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT);

    memory::Buffer<T> d_in(n_);
    memory::Buffer<T> d_out(n_);
    d_in.copy_from(h_in.data(), n_);

    memory::Buffer<T> d_work(n_);
    d_work.fill(T{0});

    CUSPARSE_CHECK(cusparseXcsrsm2_analysis(
        ctx.handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        n_, n_, 1,
        descr, buffer_.data(), d_work.data(),
        d_work.data(), d_work.data(),
        CUSPARSE_SOLVE_POLICY_NO_LEVEL,
        CUSPARSE_SOLVE_GROUP_ALG_DEFAULT));

    d_out.fill(T{0});

    d_out.copy_from(h_in.data(), n_);

    d_out.copy_to(h_out.data(), n_);

    out.copy_from(h_out.data(), n_);

    cusparseDestroyMatDescr(descr);
}

template class ILUPreconditionerImpl<double>;
template class ILUPreconditionerImpl<float>;

template<typename T>
class ILUPreconditioner : public Preconditioner<T> {
public:
    ILUPreconditioner() = default;

    void setup(const SparseMatrix<T>& A) override {
        impl_.setup(A);
    }

    void apply(const T* in, T* out) override {
        impl_.apply(in, out);
    }

    void apply(const memory::Buffer<T>& in, memory::Buffer<T>& out) override {
        impl_.apply(in, out);
    }

    double fill_in_ratio() const { return impl_.fill_in_ratio(); }

private:
    ILUPreconditionerImpl<T> impl_;
};

template class ILUPreconditioner<double>;
template class ILUPreconditioner<float>;

}
