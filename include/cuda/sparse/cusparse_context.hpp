/**
 * @file cusparse_context.hpp
 * @brief cuSPARSE context management
 * @defgroup cusparse_context cuSPARSE Context
 * @ingroup sparse
 *
 * Provides RAII wrapper for cuSPARSE library initialization and handle management.
 *
 * @note Automatically initializes cuSPARSE on construction
 * @note Thread-safe handle management
 */

#pragma once

#include <cusparse.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

/**
 * @brief Check cuSPARSE call and throw on error
 * @param expr cuSPARSE API call
 *
 * @def CUSPARSE_CHECK
 * @ingroup cusparse_context
 */
#define CUSPARSE_CHECK(expr)                                                    \
    do {                                                                        \
        cusparseStatus_t status = (expr);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                \
            throw std::runtime_error("cuSPARSE error: " +                       \
                std::to_string(static_cast<int>(status)));                      \
        }                                                                       \
    } while (0)

namespace nova::sparse::detail {

/**
 * @brief RAII wrapper for cuSPARSE handle
 * @class CusparseContext
 * @ingroup cusparse_context
 */
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
