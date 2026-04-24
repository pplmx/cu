/**
 * @file activation_buffer.cpp
 * @brief Activation buffer implementation
 */

#include "cuda/pipeline/activation_buffer.h"

namespace cuda::pipeline {

ActivationBuffer::ActivationBuffer(int device, size_t capacity)
    : device_(device),
      capacity_(capacity),
      ping_is_active_(true) {

    cudaSetDevice(device_);

    cudaMalloc(&ping_, capacity_);
    cudaMalloc(&pong_, capacity_);
}

ActivationBuffer::~ActivationBuffer() {
    cudaSetDevice(device_);
    cudaFree(ping_);
    cudaFree(pong_);
}

void* ActivationBuffer::active_buffer() const {
    return ping_is_active_ ? ping_ : pong_;
}

void* ActivationBuffer::inactive_buffer() const {
    return ping_is_active_ ? pong_ : ping_;
}

void ActivationBuffer::swap() {
    ping_is_active_ = !ping_is_active_;
}

bool ActivationBuffer::is_ping_active() const {
    return ping_is_active_;
}

size_t ActivationBuffer::capacity() const {
    return capacity_;
}

int ActivationBuffer::device() const {
    return device_;
}

}  // namespace cuda::pipeline
