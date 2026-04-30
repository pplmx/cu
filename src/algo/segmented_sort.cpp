#include "cuda/algo/segmented_sort.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace cuda::algo::segmented {

static SegmentedSortConfig g_config;

void set_config(const SegmentedSortConfig& config) {
    g_config = config;
}

SegmentedSortConfig get_config() {
    return g_config;
}

template <typename T>
void sort_by_key(const T* keys, const int* segment_ids, T* out_keys, int* out_segments,
                 size_t count, size_t num_segments, cudaStream_t stream) {
    if (stream) {
        thrust::cuda::par.on(stream);
    }

    thrust::device_ptr<const T> d_keys(keys);
    thrust::device_ptr<const int> d_segments(segment_ids);
    thrust::device_ptr<T> d_out_keys(out_keys);
    thrust::device_ptr<int> d_out_segments(out_segments);

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_keys, d_segments));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_keys + count, d_segments + count));

    if (g_config.stable) {
        thrust::stable_sort_by_key(thrust::device, zip_begin, zip_end, d_out_keys);
    } else {
        thrust::sort_by_key(thrust::device, zip_begin, zip_end, d_out_keys);
    }

    thrust::copy(thrust::device, d_segments, d_segments + count, d_out_segments);
}

template <typename T>
void sort_by_key_inplace(T* keys, int* segment_ids, size_t count,
                         size_t num_segments, cudaStream_t stream) {
    thrust::device_ptr<T> d_keys(keys);
    thrust::device_ptr<int> d_segments(segment_ids);

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_keys, d_segments));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_keys + count, d_segments + count));

    if (g_config.stable) {
        thrust::stable_sort_by_key(thrust::device, zip_begin, zip_end);
    } else {
        thrust::sort_by_key(thrust::device, zip_begin, zip_end);
    }
}

template <typename Key, typename Value>
void sort_pairs_by_key(const Key* keys, const Value* values, const int* segment_ids,
                       Key* out_keys, Value* out_values, int* out_segments,
                       size_t count, size_t num_segments, cudaStream_t stream) {
    if (stream) {
        thrust::cuda::par.on(stream);
    }

    thrust::device_ptr<const Key> d_keys(keys);
    thrust::device_ptr<const Value> d_values(values);
    thrust::device_ptr<const int> d_segments(segment_ids);
    thrust::device_ptr<Key> d_out_keys(out_keys);
    thrust::device_ptr<Value> d_out_values(out_values);
    thrust::device_ptr<int> d_out_segments(out_segments);

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_keys, d_segments));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(d_keys + count, d_segments + count));
    auto values_begin = thrust::make_zip_iterator(thrust::make_tuple(d_values, d_out_segments));

    if (g_config.stable) {
        thrust::stable_sort_by_key(thrust::device, zip_begin, zip_end, values_begin);
    } else {
        thrust::sort_by_key(thrust::device, zip_begin, zip_end, values_begin);
    }

    thrust::copy(thrust::device, d_segments, d_segments + count, d_out_segments);
}

template void sort_by_key<float>(const float*, const int*, float*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<double>(const double*, const int*, double*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<int>(const int*, const int*, int*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<int64_t>(const int64_t*, const int*, int64_t*, int*, size_t, size_t, cudaStream_t);

template void sort_by_key_inplace<float>(float*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key_inplace<double>(double*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key_inplace<int>(int*, int*, size_t, size_t, cudaStream_t);

template void sort_pairs_by_key<float, float>(const float*, const float*, const int*, float*, float*, int*, size_t, size_t, cudaStream_t);
template void sort_pairs_by_key<int, float>(const int*, const float*, const int*, int*, float*, int*, size_t, size_t, cudaStream_t);

}  // namespace cuda::algo::segmented
