#pragma once

#include "cuda/kernel/reduce.h"
#include "cuda/algo/reduce.h"

using cuda::kernel::ReduceOp;
using cuda::algo::reduce_sum;
using cuda::algo::reduce_sum_optimized;
using cuda::algo::reduce_max;
using cuda::algo::reduce_min;

// Backward compatibility aliases
template<typename T>
T reduceSum(const T* d_input, size_t size) {
    return reduce_sum(d_input, size);
}

template<typename T>
T reduceSumOptimized(const T* d_input, size_t size) {
    return reduce_sum_optimized(d_input, size);
}

template<typename T>
T reduceMax(const T* d_input, size_t size) {
    return reduce_max(d_input, size);
}

template<typename T>
T reduceMin(const T* d_input, size_t size) {
    return reduce_min(d_input, size);
}
