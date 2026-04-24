#pragma once

/**
 * @file tensor_parallel_layers.h
 * @brief Tensor-parallel layer patterns for transformers
 *
 * Implements ColumnParallelLayer and RowParallelLayer patterns
 * used in transformer architectures.
 */

#include "cuda/neural/tensor_parallel_matmul.h"
#include "cuda/nccl/nccl_all_reduce.h"

#include <memory>
#include <vector>

namespace cuda::neural {

/**
 * @class ColumnParallelLayer
 * @brief Column-parallel transformer layer
 *
 * Splits weights along output dimension. Used for QKV projection
 * in transformer attention layers.
 *
 * Input: [batch, seq, hidden_dim]
 * Output: [batch, seq, 3 * hidden_dim / tp_degree]
 */
class ColumnParallelLayer {
public:
    /**
     * @brief Construct column-parallel layer
     * @param ctx NCCL context
     * @param hidden_dim Model hidden dimension
     * @param tp_degree Tensor parallelism degree
     */
    ColumnParallelLayer(
        ::cuda::nccl::NcclContext& ctx,
        int hidden_dim,
        int tp_degree);

    // Non-copyable
    ColumnParallelLayer(const ColumnParallelLayer&) = delete;
    ColumnParallelLayer& operator=(const ColumnParallelLayer&) = delete;

    /**
     * @brief Forward pass
     * @param input Input tensor [batch * seq x hidden_dim]
     * @param output Output tensor [batch * seq x 3 * hidden_dim / tp_degree]
     * @param batch Batch size
     * @param seq Sequence length
     */
    void forward(
        const float* input,
        float* output,
        int batch,
        int seq);

    /**
     * @brief Get hidden dimension
     */
    [[nodiscard]] int hidden_dim() const;

    /**
     * @brief Get TP degree
     */
    [[nodiscard]] int tp_degree() const;

private:
    int hidden_dim_;
    int tp_degree_;
    std::unique_ptr<TensorParallelMatmul> q_proj_;
    std::unique_ptr<TensorParallelMatmul> k_proj_;
    std::unique_ptr<TensorParallelMatmul> v_proj_;
};

/**
 * @class RowParallelLayer
 * @brief Row-parallel transformer layer
 *
 * Splits weights along input dimension. Used for output projection
 * in transformer feedforward layers.
 *
 * Input: [batch, seq, hidden_dim / tp_degree]
 * Output: [batch, seq, hidden_dim]
 */
class RowParallelLayer {
public:
    /**
     * @brief Construct row-parallel layer
     * @param ctx NCCL context
     * @param hidden_dim Model hidden dimension
     * @param tp_degree Tensor parallelism degree
     */
    RowParallelLayer(
        ::cuda::nccl::NcclContext& ctx,
        int hidden_dim,
        int tp_degree);

    // Non-copyable
    RowParallelLayer(const RowParallelLayer&) = delete;
    RowParallelLayer& operator=(const RowParallelLayer&) = delete;

    /**
     * @brief Forward pass
     * @param input Input tensor [batch * seq x hidden_dim / tp_degree]
     * @param output Output tensor [batch * seq x hidden_dim]
     * @param batch Batch size
     * @param seq Sequence length
     */
    void forward(
        const float* input,
        float* output,
        int batch,
        int seq);

    /**
     * @brief Get hidden dimension
     */
    [[nodiscard]] int hidden_dim() const;

    /**
     * @brief Get TP degree
     */
    [[nodiscard]] int tp_degree() const;

private:
    int hidden_dim_;
    int tp_degree_;
    std::unique_ptr<TensorParallelMatmul> matmul_;
    ::cuda::nccl::NcclAllReduce reducer_;
};

/**
 * @class TensorParallelMLP
 * @brief Tensor-parallel MLP block
 *
 * Combines ColumnParallelLayer and RowParallelLayer for
 * transformer MLP (FFN) blocks.
 */
class TensorParallelMLP {
public:
    TensorParallelMLP(
        ::cuda::nccl::NcclContext& ctx,
        int hidden_dim,
        int intermediate_size,
        int tp_degree);

    void forward(
        const float* input,
        float* output,
        int batch,
        int seq);

private:
    int hidden_dim_;
    int tp_degree_;
    std::unique_ptr<ColumnParallelLayer> gate_proj_;
    std::unique_ptr<RowParallelLayer> up_proj_;
    std::unique_ptr<RowParallelLayer> down_proj_;
};

}  // namespace cuda::neural
