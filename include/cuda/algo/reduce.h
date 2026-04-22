#pragma once

#include <cstddef>

#include "cuda/memory/buffer.h"

namespace cuda::algo {

    template <typename T>
    auto reduce_sum(const T* input, size_t size) -> T;

    template <typename T>
    auto reduce_sum_optimized(const T* input, size_t size) -> T;

    template <typename T>
    auto reduce_max(const T* input, size_t size) -> T;

    template <typename T>
    auto reduce_min(const T* input, size_t size) -> T;

}  // namespace cuda::algo
