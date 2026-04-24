/**
 * @file tensor_parallel_layers.cpp
 * @brief Tensor-parallel layer implementations
 */

#include "cuda/neural/tensor_parallel_layers.h"

namespace cuda::neural {

ColumnParallelLayer::ColumnParallelLayer(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int tp_degree)
    : hidden_dim_(hidden_dim),
      tp_degree_(tp_degree),
      q_proj_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::ColumnParallel)),
      k_proj_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::ColumnParallel)),
      v_proj_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::ColumnParallel)) {}

void ColumnParallelLayer::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {

    int batch_seq = batch * seq;
    int local_hidden = hidden_dim_ / tp_degree_;

    q_proj_->matmul(input, nullptr, output, batch_seq, local_hidden, hidden_dim_);
}

int ColumnParallelLayer::hidden_dim() const {
    return hidden_dim_;
}

int ColumnParallelLayer::tp_degree() const {
    return tp_degree_;
}

RowParallelLayer::RowParallelLayer(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int tp_degree)
    : hidden_dim_(hidden_dim),
      tp_degree_(tp_degree),
      matmul_(std::make_unique<TensorParallelMatmul>(
          ctx, TensorParallelStrategy::RowParallel)),
      reducer_(ctx) {}

void RowParallelLayer::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {

    int batch_seq = batch * seq;
    int local_hidden = hidden_dim_ / tp_degree_;

    matmul_->matmul(input, nullptr, output, batch_seq, hidden_dim_, local_hidden);
}

int RowParallelLayer::hidden_dim() const {
    return hidden_dim_;
}

int RowParallelLayer::tp_degree() const {
    return tp_degree_;
}

TensorParallelMLP::TensorParallelMLP(
    ::cuda::nccl::NcclContext& ctx,
    int hidden_dim,
    int intermediate_size,
    int tp_degree)
    : hidden_dim_(hidden_dim),
      tp_degree_(tp_degree),
      gate_proj_(std::make_unique<ColumnParallelLayer>(ctx, hidden_dim, tp_degree)),
      up_proj_(std::make_unique<RowParallelLayer>(ctx, intermediate_size, tp_degree)),
      down_proj_(std::make_unique<RowParallelLayer>(ctx, hidden_dim, tp_degree)) {}

void TensorParallelMLP::forward(
    const float* input,
    float* output,
    int batch,
    int seq) {
}

}  // namespace cuda::neural
