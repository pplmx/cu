#pragma once

#include <nvbench/nvbench.h>

#define NOVA_BENCHMARK_KERNEL(name, func, args) \
    NVBENCH_BENCH(name) \
        .add_device_memory(#name "_data", args)

#define NOVA_BENCHMARK_MEMORY_BANDWIDTH(name, size, iterations) \
    NVBENCH_BENCH(name) \
        .set_timeout_ms(5000) \
        .add_float64_axis("Bandwidth_GB_s", {0.0})

namespace cuda::benchmark {

template <typename T>
void memory_bandwidth_benchmark(nvbench::state& state, T* d_data, size_t size) {
    state.collect_keeps_bandwidth_throughput_true(
        size * sizeof(T),
        nvbench::type_hash<T>(),
        [](nvbench::launch& launch) {});
}

template <typename T>
void compute_throughput_benchmark(nvbench::state& state, T* d_data, size_t size) {
    state.add_float64_metric("GFLOPS");
}

}  // namespace cuda::benchmark
