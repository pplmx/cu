/**
 * @file nccl_barrier.cpp
 * @brief NCCL barrier implementation
 */

#include "cuda/nccl/nccl_barrier.h"
#include "cuda/nccl/nccl_error.h"

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace cuda::nccl {

NcclBarrier::NcclBarrier(NcclContext& ctx)
    : NcclCollective(ctx) {}

NcclResult NcclBarrier::barrier_async(cudaStream_t stream) {
#ifndef NOVA_NCCL_ENABLED
    return NcclResult{.code = ncclInternalError,
                      .error_message = "NCCL not enabled"};
#else
    if (!has_nccl()) {
        return NcclResult{.code = ncclInternalError,
                          .error_message = "NCCL context not initialized"};
    }

    return safe_nccl_call(
        [&]() {
            ncclComm_t comm = get_comm(0);
            int dummy = 0;
            return ncclAllReduce(&dummy, &dummy, 1, ncclInt, ncclMin, comm, stream);
        },
        get_comm(0),
        30000);
#endif
}

NcclResult NcclBarrier::barrier_async(int device, cudaStream_t stream) {
#ifndef NOVA_NCCL_ENABLED
    return NcclResult{.code = ncclInternalError,
                      .error_message = "NCCL not enabled"};
#else
    if (!has_nccl()) {
        return NcclResult{.code = ncclInternalError,
                          .error_message = "NCCL context not initialized"};
    }

    return safe_nccl_call(
        [&]() {
            ncclComm_t comm = get_comm(device);
            int dummy = 0;
            return ncclAllReduce(&dummy, &dummy, 1, ncclInt, ncclMin, comm, stream);
        },
        get_comm(device),
        30000);
#endif
}

NcclResult NcclBarrier::barrier(cudaStream_t stream) {
    auto result = barrier_async(stream);
    if (result.ok()) {
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            result.code = ncclSystemError;
            result.error_message = cudaGetErrorString(sync_err);
        }
    }
    return result;
}

NcclResult NcclBarrier::barrier(int device, cudaStream_t stream) {
    auto result = barrier_async(device, stream);
    if (result.ok()) {
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            result.code = ncclSystemError;
            result.error_message = cudaGetErrorString(sync_err);
        }
    }
    return result;
}

}  // namespace cuda::nccl
