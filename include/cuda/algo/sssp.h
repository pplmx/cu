#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <limits>

#include "cuda/graph/csr_graph.h"
#include "cuda/memory/buffer.h"

namespace cuda::algo::sssp {

constexpr float INF = std::numeric_limits<float>::infinity();

template <typename Weight>
void delta_stepping(const graph::CSRGraph& graph, int source, Weight* distances,
                    Weight delta = Weight{1.0}, cudaStream_t stream = nullptr);

template <typename Weight>
void bellman_ford(const graph::CSRGraph& graph, int source, Weight* distances,
                  cudaStream_t stream = nullptr);

template <typename Weight>
memory::Buffer<Weight> compute_distances(const graph::CSRGraph& graph, int source,
                                         Weight delta = Weight{1.0},
                                         cudaStream_t stream = nullptr);

struct SSSPResult {
    memory::Buffer<float> distances;
    int num_vertices;
    bool converged;
    int iterations;
};

SSSPResult run(const graph::CSRGraph& graph, int source,
               float delta = 1.0f, cudaStream_t stream = nullptr);

struct SSSPConfig {
    float default_delta = 1.0f;
    int max_iterations = 1000;
    float convergence_threshold = 1e-6f;
    bool use_delta_stepping = true;
};

void set_config(const SSSPConfig& config);
SSSPConfig get_config();

}  // namespace cuda::algo::sssp
