#include "cuda/algo/segmented_sort.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/functional.h>

namespace cuda::algo::segmented {

static SegmentedSortConfig g_config;

void set_config(const SegmentedSortConfig& config) {
    g_config = config;
}

SegmentedSortConfig get_config() {
    return g_config;
}

namespace {
    template <typename T>
    struct SegmentComparator {
        const int* segments;
        SegmentComparator(const int* segs) : segments(segs) {}
        __device__ bool operator()(int a, int b) const {
            return segments[a] < segments[b];
        }
    };
}

template <typename T>
void sort_by_key(const T* keys, const int* segment_ids, T* out_keys, int* out_segments,
                 size_t count, size_t num_segments, cudaStream_t stream) {
    T* d_keys_sorted;
    T* d_keys_original;
    int* d_segments_original;
    int* d_indices;
    cudaMalloc(&d_keys_sorted, count * sizeof(T));
    cudaMalloc(&d_keys_original, count * sizeof(T));
    cudaMalloc(&d_segments_original, count * sizeof(int));
    cudaMalloc(&d_indices, count * sizeof(int));
    cudaMemcpy(d_keys_original, keys, count * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segments_original, segment_ids, count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys_sorted, keys, count * sizeof(T), cudaMemcpyHostToDevice);

    thrust::device_ptr<int> d_indices_ptr(d_indices);
    thrust::sequence(d_indices_ptr, d_indices_ptr + count);

    thrust::device_ptr<T> d_keys_sorted_ptr(d_keys_sorted);

    if (g_config.stable) {
        thrust::stable_sort_by_key(d_keys_sorted_ptr, d_keys_sorted_ptr + count, d_indices_ptr);
    } else {
        thrust::sort_by_key(d_keys_sorted_ptr, d_keys_sorted_ptr + count, d_indices_ptr);
    }

    thrust::device_ptr<T> d_out_keys_ptr(out_keys);
    thrust::device_ptr<int> d_out_segments_ptr(out_segments);
    thrust::gather(d_indices_ptr, d_indices_ptr + count,
                   thrust::device_ptr<const T>(d_keys_original), d_out_keys_ptr);
    thrust::gather(d_indices_ptr, d_indices_ptr + count,
                   thrust::device_ptr<const int>(d_segments_original), d_out_segments_ptr);

    cudaFree(d_keys_sorted);
    cudaFree(d_keys_original);
    cudaFree(d_segments_original);
    cudaFree(d_indices);
}

template <typename T>
void sort_by_key_inplace(T* keys, int* segment_ids, size_t count,
                         size_t num_segments, cudaStream_t stream) {
    thrust::device_ptr<T> d_keys(keys);
    thrust::device_ptr<int> d_segments(segment_ids);

    auto keys_begin = thrust::make_zip_iterator(thrust::make_tuple(d_keys, d_segments));
    auto keys_end = keys_begin + count;

    if (g_config.stable) {
        thrust::stable_sort_by_key(keys_begin, keys_end, d_keys);
    } else {
        thrust::sort_by_key(keys_begin, keys_end, d_keys);
    }
}

template void sort_by_key<float>(const float*, const int*, float*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<double>(const double*, const int*, double*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<int>(const int*, const int*, int*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key<int64_t>(const int64_t*, const int*, int64_t*, int*, size_t, size_t, cudaStream_t);

template void sort_by_key_inplace<float>(float*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key_inplace<double>(double*, int*, size_t, size_t, cudaStream_t);
template void sort_by_key_inplace<int>(int*, int*, size_t, size_t, cudaStream_t);

}  // namespace cuda::algo::segmented
