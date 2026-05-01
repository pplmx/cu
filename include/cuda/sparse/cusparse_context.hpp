#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUSPARSE_CHECK(expr)                                                    \
    do {                                                                        \
        cusparseStatus_t status = (expr);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                \
            throw std::runtime_error("cuSPARSE error: " +                       \
                std::to_string(static_cast<int>(status)));                      \
        }                                                                       \
    } while (0)

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
