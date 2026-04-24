/**
 * @file nccl_context.cpp
 * @brief NCCL context implementation
 *
 * Provides communicator management for multi-GPU collective operations.
 * Full implementation in plan 13-02.
 */

#include "cuda/nccl/nccl_context.h"

namespace cuda::nccl {

void NcclContext::initialize() {
    // Full implementation in plan 13-02
    initialized_ = true;
}

void NcclContext::initialize(const NcclContextConfig& config) {
    (void)config;
    // Full implementation in plan 13-02
    initialized_ = true;
}

NcclContext::NcclContext(const NcclContextConfig& config) {
    initialize(config);
}

NcclContext::NcclContext(NcclContext&& other) noexcept
    : device_count_(other.device_count_),
      communicators_(std::move(other.communicators_)),
      streams_(std::move(other.streams_)),
      device_ids_(std::move(other.device_ids_)),
      initialized_(other.initialized_) {
    other.initialized_ = false;
    other.device_count_ = 0;
}

NcclContext& NcclContext::operator=(NcclContext&& other) noexcept {
    if (this != &other) {
        destroy();
        device_count_ = other.device_count_;
        communicators_ = std::move(other.communicators_);
        streams_ = std::move(other.streams_);
        device_ids_ = std::move(other.device_ids_);
        initialized_ = other.initialized_;
        other.initialized_ = false;
        other.device_count_ = 0;
    }
    return *this;
}

NcclContext::~NcclContext() {
    destroy();
}

void NcclContext::destroy() {
    if (!initialized_) {
        return;
    }

    for (auto& stream : streams_) {
        if (stream != cudaStreamDefault && stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
    streams_.clear();

#ifdef NOVA_NCCL_ENABLED
    for (auto& comm : communicators_) {
        if (comm != nullptr) {
            ncclCommDestroy(comm);
        }
    }
#endif

    communicators_.clear();
    device_count_ = 0;
    device_ids_.clear();
    initialized_ = false;
}

ncclComm_t NcclContext::get_comm(int device) const {
    if (!initialized_) {
#ifdef NOVA_NCCL_ENABLED
        throw NcclException("NcclContext not initialized", ncclInvalidArgument,
                            "get_comm", __FILE__, __LINE__);
#else
        throw std::runtime_error("NcclContext not initialized");
#endif
    }

    auto it = std::find(device_ids_.begin(), device_ids_.end(), device);
    if (it == device_ids_.end()) {
#ifdef NOVA_NCCL_ENABLED
        throw NcclException("Device not in NCCL group", ncclInvalidArgument,
                            "get_comm", __FILE__, __LINE__);
#else
        throw std::runtime_error("Device not in NCCL group");
#endif
    }

    return communicators_[std::distance(device_ids_.begin(), it)];
}

cudaStream_t NcclContext::get_stream(int device) const {
    auto it = std::find(device_ids_.begin(), device_ids_.end(), device);
    if (it == device_ids_.end()) {
#ifdef NOVA_NCCL_ENABLED
        throw NcclException("Device not in NCCL group", ncclInvalidArgument,
                            "get_stream", __FILE__, __LINE__);
#else
        throw std::runtime_error("Device not in NCCL group");
#endif
    }

    return streams_[std::distance(device_ids_.begin(), it)];
}

}  // namespace cuda::nccl
