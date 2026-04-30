# v2.9 Architecture Refactor: Unified Sparse + cuSPARSE Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the sparse module to use GPU memory (Buffer<T>), integrate cuSPARSE for acceleration, and unify CSR abstractions while maintaining backward compatibility.

**Architecture:** New `SparseMatrix<T>` class uses `cuda::memory::Buffer<T>` for device storage. cuSPARSE context singleton wraps handle management. Solvers use cuSOLVER sparse routines. Old `SparseMatrixCSR<T>` marked deprecated but functional.

**Tech Stack:** C++23, CUDA 20, cuSPARSE, cuSOLVER, Google Test

---

## File Structure

```
include/cuda/sparse/
├── matrix.hpp              # NEW: Unified SparseMatrix<T> with Buffer<T>
├── cusparse_context.hpp    # NEW: cuSPARSE handle wrapper
├── krylov.hpp              # MODIFY: Add GPU solver implementations

src/cuda/sparse/
├── sparse_ops.cu           # NEW: GPU implementations for operations
├── cusparse_context.cpp    # NEW: cuSPARSE initialization
└── krylov_gpu.cpp          # NEW: cuSOLVER-based solvers

tests/sparse/
├── matrix_test.cpp         # NEW: SparseMatrix<T> tests
├── cusparse_test.cpp       # NEW: cuSPARSE context tests
└── krylov_test.cpp         # MODIFY: Add GPU solver validation tests

CMakeLists.txt (root)       # MODIFY: Add cuSPARSE/cuSOLVER linking, sparse source files
CMakeLists.txt (tests)      # MODIFY: Add new test executables
```

---

## Phase 83: SparseMatrix with Buffer<T>

### Task 83.1: Create SparseMatrix<T> Header

**Files:**
- Create: `include/cuda/sparse/matrix.hpp`

- [ ] **Step 1: Write the header with SparseMatrix<T> class**

```cpp
#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "cuda/memory/buffer.h"

namespace nova::sparse {

enum class SparseFormat { CSR, CSC, ELL, SELL, HYB };

template<typename T>
class SparseMatrix {
public:
    SparseMatrix() = default;

    SparseMatrix(int num_rows, int num_cols, int nnz)
        : values_(nnz)
        , row_offsets_(num_rows + 1)
        , col_indices_(nnz)
        , num_rows_(num_rows)
        , num_cols_(num_cols) {}

    static std::optional<SparseMatrix> FromDense(const T* dense, int rows, int cols,
                                                  float sparsity_threshold = 0.0f);

    static SparseMatrix FromHostData(std::vector<T> values,
                                      std::vector<int> row_offsets,
                                      std::vector<int> col_indices,
                                      int num_rows, int num_cols);

    int rows() const { return num_rows_; }
    int cols() const { return num_cols_; }
    int nnz() const { return static_cast<int>(values_.size()); }

    T* values() { return values_.data(); }
    int* row_offsets() { return row_offsets_.data(); }
    int* col_indices() { return col_indices_.data(); }

    const T* values() const { return values_.data(); }
    const int* row_offsets() const { return row_offsets_.data(); }
    const int* col_indices() const { return col_indices_.data(); }

    memory::Buffer<T>& values_buffer() { return values_; }
    memory::Buffer<int>& row_offsets_buffer() { return row_offsets_; }
    memory::Buffer<int>& col_indices_buffer() { return col_indices_; }

private:
    memory::Buffer<T> values_;
    memory::Buffer<int> row_offsets_;
    memory::Buffer<int> col_indices_;
    int num_rows_ = 0;
    int num_cols_ = 0;
};

}  // namespace nova::sparse
```

- [ ] **Step 2: Add FromDense implementation (inline in header)**

```cpp
template<typename T>
std::optional<SparseMatrix<T>> SparseMatrix<T>::FromDense(const T* dense,
                                                          int rows, int cols,
                                                          float threshold) {
    std::vector<T> values;
    std::vector<int> row_offsets(1, 0);
    std::vector<int> col_indices;

    for (int i = 0; i < rows; ++i) {
        int row_nnz = 0;
        for (int j = 0; j < cols; ++j) {
            T val = dense[i * cols + j];
            if (val != T{0} && std::abs(val) > threshold) {
                values.push_back(val);
                col_indices.push_back(j);
                ++row_nnz;
            }
        }
        row_offsets.push_back(static_cast<int>(values.size()));
    }

    SparseMatrix<T> result(rows, cols, static_cast<int>(values.size()));
    result.values_buffer().copy_from(values.data(), values.size());
    result.row_offsets_buffer().copy_from(row_offsets.data(), row_offsets.size());
    result.col_indices_buffer().copy_from(col_indices.data(), col_indices.size());

    return result;
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>::FromHostData(std::vector<T> values,
                                               std::vector<int> row_offsets,
                                               std::vector<int> col_indices,
                                               int num_rows, int num_cols) {
    SparseMatrix<T> result(num_rows, num_cols, static_cast<int>(values.size()));
    result.values_buffer().copy_from(values.data(), values.size());
    result.row_offsets_buffer().copy_from(row_offsets.data(), row_offsets.size());
    result.col_indices_buffer().copy_from(col_indices.data(), col_indices.size());
    return result;
}
```

- [ ] **Step 3: Commit**

```bash
git add include/cuda/sparse/matrix.hpp
git commit -m "feat(sparse): add SparseMatrix<T> with cuda::memory::Buffer<T>"
```

### Task 83.2: Create SparseMatrix Tests

**Files:**
- Create: `tests/sparse/matrix_test.cpp`

- [ ] **Step 1: Write the test file**

```cpp
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

TEST(SparseMatrixTest, FromHostData) {
    std::vector<float> values = {4.0f, 1.0f, 1.0f, 3.0f};
    std::vector<int> row_offsets = {0, 2, 4};
    std::vector<int> col_indices = {0, 1, 0, 1};

    auto matrix = SparseMatrix<float>::FromHostData(values, row_offsets, col_indices, 2, 2);

    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_EQ(matrix.nnz(), 4);
}

TEST(SparseMatrixTest, BufferOwnership) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f};
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};

    auto matrix = SparseMatrix<float>::FromHostData(values, row_offsets, col_indices, 3, 3);

    EXPECT_NE(matrix.values(), nullptr);
    EXPECT_NE(matrix.row_offsets(), nullptr);
    EXPECT_NE(matrix.col_indices(), nullptr);
}

}  // namespace nova::sparse::test
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd build && ctest -R SparseMatrixTest -V`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/sparse/matrix_test.cpp
git commit -m "test(sparse): add SparseMatrix<T> tests"
```

---

## Phase 84: cuSPARSE Integration

### Task 84.1: Create cuSPARSE Context

**Files:**
- Create: `include/cuda/sparse/cusparse_context.hpp`
- Create: `src/cuda/sparse/cusparse_context.cpp`

- [ ] **Step 1: Write the header**

```cpp
#pragma once

#include <cusparse.h>

namespace nova::sparse::detail {

class CusparseContext {
public:
    static CusparseContext& get();

    cusparseHandle_t handle() { return handle_; }
    cudaStream_t stream() { return stream_; }
    void set_stream(cudaStream_t stream);

    CusparseContext(const CusparseContext&) = delete;
    CusparseContext& operator=(const CusparseContext&) = delete;
    CusparseContext(CusparseContext&&) = delete;
    CusparseContext& operator=(CusparseContext&&) = delete;

private:
    CusparseContext();
    ~CusparseContext();

    cusparseHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

}  // namespace nova::sparse::detail
```

- [ ] **Step 2: Write the implementation**

```cpp
#include "cuda/sparse/cusparse_context.hpp"
#include "cuda/device/error.h"

namespace nova::sparse::detail {

CusparseContext::CusparseContext() {
    CUSPARSE_CHECK(cusparseCreate(&handle_));
}

CusparseContext::~CusparseContext() {
    if (handle_) {
        cusparseDestroy(handle_);
        handle_ = nullptr;
    }
}

CusparseContext& CusparseContext::get() {
    static CusparseContext instance;
    return instance;
}

void CusparseContext::set_stream(cudaStream_t stream) {
    stream_ = stream;
    CUSPARSE_CHECK(cusparseSetStream(handle_, stream));
}

}  // namespace nova::sparse::detail
```

- [ ] **Step 3: Create error checking macro**

Add to `include/cuda/sparse/cusparse_context.hpp`:

```cpp
#define CUSPARSE_CHECK(expr)                                                    \
    do {                                                                        \
        cusparseStatus_t status = (expr);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                \
            throw std::runtime_error("cuSPARSE error: " +                       \
                std::to_string(static_cast<int>(status)));                      \
        }                                                                       \
    } while (0)
```

- [ ] **Step 4: Commit**

```bash
git add include/cuda/sparse/cusparse_context.hpp src/cuda/sparse/cusparse_context.cpp
git commit -m "feat(sparse): add cuSPARSE context singleton"
```

### Task 84.2: Implement GPU SpMV

**Files:**
- Create: `src/cuda/sparse/sparse_ops.cu`
- Modify: `include/cuda/sparse/matrix.hpp` (add spmv function)

- [ ] **Step 1: Add spmv declaration to matrix.hpp**

Add after the SparseMatrix class:

```cpp
namespace nova::sparse {

template<typename T>
void spmv(const SparseMatrix<T>& A, const T* x, T* y);

template<typename T>
void spmv_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream);

}  // namespace nova::sparse
```

- [ ] **Step 2: Write the CUDA SpMV implementation**

```cpp
#include "cuda/sparse/matrix.hpp"
#include "cuda/sparse/cusparse_context.hpp"
#include <cusparse.h>

namespace nova::sparse {

namespace detail {

template<typename T>
struct CusparseTraits;

template<>
struct CusparseTraits<float> {
    static constexpr cusparseDataType_t type = CUDA_R_32F;
};

template<>
struct CusparseTraits<double> {
    static constexpr cusparseDataType_t type = CUDA_R_64F;
};

template<typename T>
void spmv_impl(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
    auto& ctx = CusparseContext::get();
    if (stream) {
        ctx.set_stream(stream);
    }

    cusparseSpMatDescr_t mat_desc;
    cusparseDnVecDescr_t vec_x_desc;
    cusparseDnVecDescr_t vec_y_desc;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_desc,
        A.rows(), A.cols(), A.nnz(),
        const_cast<int*>(A.row_offsets()), const_cast<int*>(A.col_indices()),
        const_cast<T*>(A.values()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CusparseTraits<T>::type
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_x_desc,
        A.cols(), const_cast<T*>(x),
        CusparseTraits<T>::type
    ));

    CUSPARSE_CHECK(cusparseCreateDnVec(
        &vec_y_desc,
        A.rows(), y,
        CusparseTraits<T>::type
    ));

    T alpha = T{1};
    T beta = T{0};

    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc,
        CusparseTraits<T>::type,
        CUSPARSE_MV_ALG_DEFAULT,
        &buffer_size
    ));

    memory::Buffer<void> buffer(buffer_size);

    CUSPARSE_CHECK(cusparseSpMV(
        ctx.handle(),
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc, vec_x_desc, &beta, vec_y_desc,
        CusparseTraits<T>::type,
        CUSPARSE_MV_ALG_DEFAULT,
        buffer.data()
    ));

    CUSPARSE_CHECK(cusparseDestroySpMat(mat_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x_desc));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y_desc));
}

}  // namespace detail

template<typename T>
void spmv(const SparseMatrix<T>& A, const T* x, T* y) {
    detail::spmv_impl(A, x, y, nullptr);
}

template<typename T>
void spmv_async(const SparseMatrix<T>& A, const T* x, T* y, cudaStream_t stream) {
    detail::spmv_impl(A, x, y, stream);
}

}  // namespace nova::sparse
```

- [ ] **Step 3: Write test**

Add to `tests/sparse/matrix_test.cpp`:

```cpp
TEST(SparseMatrixTest, SpmvGPU) {
    std::vector<float> dense = {
        4.0f, 1.0f, 0.0f,
        1.0f, 3.0f, 1.0f,
        0.0f, 1.0f, 2.0f
    };
    auto matrix = SparseMatrix<float>::FromDense(dense.data(), 3, 3, 0.0f);
    ASSERT_TRUE(matrix.has_value());

    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    memory::Buffer<float> d_x(3);
    memory::Buffer<float> d_y(3);
    memory::Buffer<float> h_y(3);

    d_x.copy_from(x.data(), 3);

    spmv(*matrix, d_x.data(), d_y.data());

    h_y.copy_to(d_y.data(), 3);

    EXPECT_TRUE(approx_equal(h_y[0], 6.0f));
    EXPECT_TRUE(approx_equal(h_y[1], 9.0f));
    EXPECT_TRUE(approx_equal(h_y[2], 5.0f));
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && ctest -R SparseMatrixTest -V`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cuda/sparse/sparse_ops.cu include/cuda/sparse/matrix.hpp
git commit -m "feat(sparse): implement GPU SpMV via cuSPARSE"
```

---

## Phase 85: GPU-Accelerated Krylov Solvers

### Task 85.1: Refactor Krylov to Use GPU

**Files:**
- Modify: `include/cuda/sparse/krylov.hpp`

- [ ] **Step 1: Add GPU solver implementations**

Modify the solver classes to use cuSOLVER sparse routines:

```cpp
// Add to krylov.hpp - GPU-accelerated implementations

namespace detail {

template<typename T>
class GpuKrylovContext {
public:
    static GpuKrylovContext& get();

    void spmv(const SparseMatrix<T>& A, const T* x, T* y) {
        nova::sparse::spmv(A, x, y);
    }

    T dot(const T* a, const T* b, size_t n) {
        memory::Buffer<T> d_a(n), d_b(n), d_result(1);
        d_a.copy_from(a, n);
        d_b.copy_from(b, n);

        cublasHandle_t cublas = cuda::device::get_cublas_handle();
        T result;
        cublasDot(cublas, n, d_a.data(), 1, d_b.data(), 1, &result);

        std::vector<T> h_result(1);
        d_result.copy_to(h_result.data(), 1);
        return h_result[0];
    }

    void axpy(T a, const T* x, T* y, size_t n) {
        // Implement using CUDA kernel or cublas
        memory::Buffer<T> d_a(n), d_b(n);
        d_a.copy_from(x, n);
        d_b.copy_from(y, n);

        cublasHandle_t cublas = cuda::device::get_cublas_handle();
        cublasAxpy(cublas, n, &a, d_a.data(), 1, d_b.data(), 1);

        std::vector<T> h_b(n);
        d_b.copy_to(h_b.data(), n);
        for (size_t i = 0; i < n; ++i) {
            y[i] = h_b[i];
        }
    }

    void copy(const T* src, T* dst, size_t n) {
        memory::Buffer<T> d_src(n), d_dst(n);
        d_src.copy_from(src, n);
        d_dst.copy_from(dst, n);
        CUDA_CHECK(cudaMemcpy(d_dst.data(), d_src.data(), n * sizeof(T), cudaMemcpyDeviceToDevice));
        std::vector<T> h_dst(n);
        d_dst.copy_to(h_dst.data(), n);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = h_dst[i];
        }
    }

    void scale(T* x, T alpha, size_t n) {
        memory::Buffer<T> d_x(n);
        d_x.copy_from(x, n);

        cublasHandle_t cublas = cuda::device::get_cublas_handle();
        cublasScale(cublas, n, &alpha, d_x.data(), 1);

        std::vector<T> h_x(n);
        d_x.copy_to(h_x.data(), n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = h_x[i];
        }
    }

private:
    GpuKrylovContext() = default;
};

}  // namespace detail
```

- [ ] **Step 2: Update ConjugateGradient to use GPU**

```cpp
template<typename T>
SolverResult<T> ConjugateGradient<T>::solve(const SparseMatrix<T>& A, const T* b, T* x) {
    const int n = A.rows();
    auto& ctx = detail::GpuKrylovContext<T>::get();

    std::vector<T> r(n), p(n), Ap(n);
    std::vector<T> x_vec(n, T{0});

    // Initial residual r = b - A*x (x starts at 0, so r = b)
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
        p[i] = b[i];
    }

    T r_dot_old = ctx.dot(r.data(), r.data(), n);
    T b_norm = std::sqrt(r_dot_old);

    SolverResult<T> result;
    result.residual_history.reserve(config_.max_iterations);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Ap = A*p
        nova::sparse::spmv(A, p.data(), Ap.data());

        T p_Ap = ctx.dot(p.data(), Ap.data(), n);

        if (std::abs(p_Ap) < T{1e-12}) {
            result.error_code = SolverError::BREAKDOWN;
            break;
        }

        T alpha = r_dot_old / p_Ap;

        // x = x + alpha*p
        ctx.axpy(alpha, p.data(), x_vec.data(), n);

        // r = r - alpha*Ap
        ctx.axpy(-alpha, Ap.data(), r.data(), n);

        T r_dot_new = ctx.dot(r.data(), r.data(), n);
        T residual_norm = std::sqrt(r_dot_new);

        result.residual_history.push_back(residual_norm);

        if (residual_norm < config_.relative_tolerance * b_norm) {
            result.converged = true;
            result.iterations = iter + 1;
            result.residual_norm = residual_norm;
            result.relative_residual = residual_norm / b_norm;
            break;
        }

        T beta = r_dot_new / r_dot_old;

        // p = r + beta*p
        ctx.scale(p.data(), beta, n);
        ctx.axpy(T{1}, r.data(), p.data(), n);

        r_dot_old = r_dot_new;
    }

    if (!result.converged && result.iterations == 0) {
        result.iterations = config_.max_iterations;
        result.error_code = SolverError::MAX_ITERATIONS;
    }

    for (int i = 0; i < n; ++i) {
        x[i] = x_vec[i];
    }

    return result;
}
```

- [ ] **Step 3: Commit**

```bash
git add include/cuda/sparse/krylov.hpp
git commit -m "feat(krylov): refactor to use GPU with cuSPARSE/cuBLAS"
```

---

## Phase 86: Backward Compatibility & Integration

### Task 86.1: Deprecate Old API

**Files:**
- Modify: `include/cuda/sparse/sparse_matrix.hpp`

- [ ] **Step 1: Add deprecation attribute**

```cpp
template<typename T>
class [[deprecated("Use cuda::sparse::SparseMatrix<T> instead")]]
SparseMatrixCSR {
    // ... existing code unchanged
};
```

- [ ] **Step 2: Add conversion function**

```cpp
// In sparse_matrix.hpp

template<typename T>
SparseMatrix<T> ToSparseMatrix(const SparseMatrixCSR<T>& csr) {
    std::vector<T> values(csr.nnz());
    std::vector<int> row_offsets(csr.num_rows() + 1);
    std::vector<int> col_indices(csr.nnz());

    std::copy(csr.values(), csr.values() + csr.nnz(), values.begin());
    std::copy(csr.row_offsets(), csr.row_offsets() + csr.num_rows() + 1, row_offsets.begin());
    std::copy(csr.col_indices(), csr.col_indices() + csr.nnz(), col_indices.begin());

    return SparseMatrix<T>::FromHostData(values, row_offsets, col_indices,
                                          csr.num_rows(), csr.num_cols());
}
```

- [ ] **Step 3: Commit**

```bash
git add include/cuda/sparse/sparse_matrix.hpp
git commit -m "feat(sparse): deprecate SparseMatrixCSR, add conversion to SparseMatrix"
```

### Task 86.2: Update CMake Build

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add cuSPARSE/cuSOLVER linking**

Find the `cuda_algo` library section and add:

```cmake
set(CUDA_SPARSE_DIR ${CMAKE_SOURCE_DIR}/include/cuda/sparse)
set(SRC_CUDA_SPARSE ${CMAKE_SOURCE_DIR}/src/cuda/sparse)

set(SPARSE_SOURCES
        ${SRC_CUDA_SPARSE}/cusparse_context.cpp
        ${SRC_CUDA_SPARSE}/sparse_ops.cu
)
```

Add to the appropriate library target:

```cmake
target_link_libraries(cuda_algo INTERFACE
        CUDA::cublas
        CUDA::cusparse  # ADD THIS
        CUDA::cusolver  # ADD THIS
)
```

- [ ] **Step 2: Update tests CMakeLists.txt**

```cmake
# Add new test executables
add_executable(sparse-matrix-tests
    sparse/matrix_test.cpp
)

target_link_libraries(sparse-matrix-tests PRIVATE
    nova-lib
    GTest::gtest_main
)
gtest_discover_tests(sparse-matrix-tests)
```

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt
git commit -m "build: add cuSPARSE/cuSOLVER linking and sparse tests"
```

---

## Phase 87: Final Integration Tests

### Task 87.1: End-to-End Validation

**Files:**
- Modify: `tests/sparse/krylov_test.cpp`

- [ ] **Step 1: Add GPU vs CPU comparison test**

```cpp
TEST_F(KrylovSolverTest, GpuVsCpuConsistency) {
    const int n = 100;
    std::vector<T> dense(n * n, T{0});

    // Create SPD tridiagonal matrix
    for (int i = 0; i < n; ++i) {
        dense[i * n + i] = T{4};
        if (i > 0) dense[i * n + i - 1] = T{-1};
        if (i < n - 1) dense[i * n + i + 1] = T{-1};
    }

    // CPU solution (existing API)
    auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), n, n);
    ASSERT_TRUE(csr.has_value());

    std::vector<T> b_cpu(n, T{1});
    std::vector<T> x_cpu(n, T{0});

    SolverConfig<T> config;
    config.relative_tolerance = T{1e-8};
    config.max_iterations = 500;

    ConjugateGradient<T> solver_cpu(config);
    auto result_cpu = solver_cpu.solve(*csr, b_cpu.data(), x_cpu.data());

    // GPU solution (new API)
    auto gpu_matrix = ToSparseMatrix(*csr);
    std::vector<T> b_gpu(n, T{1});
    std::vector<T> x_gpu(n, T{0});

    memory::Buffer<T> d_b(n), d_x(n);
    d_b.copy_from(b_gpu.data(), n);

    auto gpu_solver = cuda::sparse::ConjugateGradient<T>(config);
    auto result_gpu = gpu_solver.solve(gpu_matrix, d_b.data(), d_x.data());

    // Compare solutions
    std::vector<T> x_gpu_host(n);
    d_x.copy_to(x_gpu_host.data(), n);

    EXPECT_EQ(result_cpu.converged, result_gpu.converged);
    if (result_cpu.converged && result_gpu.converged) {
        EXPECT_TRUE(approx_equal(result_cpu.iterations, result_gpu.iterations, 5));
    }

    for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(approx_equal(x_cpu[i], x_gpu_host[i], T{1e-4}));
    }
}
```

- [ ] **Step 2: Run all sparse tests**

Run: `cd build && ctest -R sparse -V`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/sparse/krylov_test.cpp
git commit -m "test(sparse): add GPU vs CPU consistency test"
```

---

## Self-Review Checklist

- [ ] All spec requirements covered by tasks
- [ ] No "TBD" or placeholder content
- [ ] File paths are exact
- [ ] Code is complete and compilable
- [ ] Tests verify the implementation
- [ ] Commits are atomic and meaningful

---

## Out of Scope (Future Work)

- Custom CUDA kernels for ELL/SELL/HYB formats
- Multi-GPU sparse operations
- Sparse format autotuning based on Roofline

---

*Plan created: 2026-05-01*
*v2.9 Architecture Refactor Implementation Plan*
