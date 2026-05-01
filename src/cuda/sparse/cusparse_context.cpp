#include "cuda/sparse/cusparse_context.hpp"
#include "cuda/device/error.h"

namespace nova::sparse::detail {

CusparseContext::CusparseContext() {
    cusparseStatus_t status = cusparseCreate(&handle_);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuSPARSE handle: " +
                                  std::to_string(static_cast<int>(status)));
    }
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
    cusparseSetStream(handle_, stream);
}

}  // namespace nova::sparse::detail
