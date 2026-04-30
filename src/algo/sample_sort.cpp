#include "cuda/algo/sample_sort.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

namespace cuda::algo::sample_sort {

static SampleSortConfig g_config;

void set_config(const SampleSortConfig& config) {
    g_config = config;
}

SampleSortConfig get_config() {
    return g_config;
}

template <typename T>
SortResult<T> sort(const T* input, size_t count, Order order, cudaStream_t stream) {
    SortResult<T> result;

    if (stream) {
        thrust::cuda::par.on(stream);
    }

    thrust::device_ptr<const T> d_input(input);
    thrust::device_ptr<T> d_output = thrust::device_malloc<T>(count);

    thrust::copy(thrust::device, d_input, d_input + count, d_output);

    if (order == Order::Ascending) {
        thrust::sort(thrust::device, d_output, d_output + count);
    } else {
        thrust::sort(thrust::device, d_output, d_output + count, thrust::greater<T>());
    }

    result.data.reset(d_output.get());
    result.data.set_size(count);
    result.actual_count = count;

    return result;
}

template <typename T>
SortResult<T> sort_large_dataset(const T* input, size_t count,
                                 Order order, size_t sample_rate,
                                 cudaStream_t stream) {
    SortResult<T> result;

    if (count <= g_config.threshold_large_dataset) {
        return sort(input, count, order, stream);
    }

    if (stream) {
        thrust::cuda::par.on(stream);
    }

    thrust::device_ptr<const T> d_input(input);

    size_t num_samples = count / sample_rate;
    thrust::device_ptr<T> d_samples = thrust::device_malloc<T>(num_samples);
    thrust::device_ptr<T> d_output = thrust::device_malloc<T>(count);

    thrust::copy(thrust::device, d_input, d_input + num_samples, d_samples);

    if (order == Order::Ascending) {
        thrust::sort(thrust::device, d_samples, d_samples + num_samples);
    } else {
        thrust::sort(thrust::device, d_samples, d_samples + num_samples, thrust::greater<T>());
    }

    thrust::copy(thrust::device, d_input, d_input + count, d_output);

    if (order == Order::Ascending) {
        thrust::stable_sort(thrust::device, d_output, d_output + count);
    } else {
        thrust::stable_sort(thrust::device, d_output, d_output + count, thrust::greater<T>());
    }

    thrust::device_free(d_samples);

    result.data.reset(d_output.get());
    result.data.set_size(count);
    result.actual_count = count;

    return result;
}

template <typename T>
void sort_inplace(T* data, size_t count, Order order, cudaStream_t stream) {
    if (stream) {
        thrust::cuda::par.on(stream);
    }

    thrust::device_ptr<T> d_data(data);

    if (order == Order::Ascending) {
        thrust::sort(thrust::device, d_data, d_data + count);
    } else {
        thrust::sort(thrust::device, d_data, d_data + count, thrust::greater<T>());
    }
}

template SortResult<float> sort<float>(const float*, size_t, Order, cudaStream_t);
template SortResult<double> sort<double>(const double*, size_t, Order, cudaStream_t);
template SortResult<int> sort<int>(const int*, size_t, Order, cudaStream_t);

template SortResult<float> sort_large_dataset<float>(const float*, size_t, Order, size_t, cudaStream_t);
template SortResult<double> sort_large_dataset<double>(const double*, size_t, Order, size_t, cudaStream_t);
template SortResult<int> sort_large_dataset<int>(const int*, size_t, Order, size_t, cudaStream_t);

template void sort_inplace<float>(float*, size_t, Order, cudaStream_t);
template void sort_inplace<double>(double*, size_t, Order, cudaStream_t);
template void sort_inplace<int>(int*, size_t, Order, cudaStream_t);

}  // namespace cuda::algo::sample_sort
