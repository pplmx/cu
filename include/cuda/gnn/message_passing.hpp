#ifndef NOVA_CUDA_GNN_MESSAGE_PASSING_HPP
#define NOVA_CUDA_GNN_MESSAGE_PASSING_HPP

#include <cuda/sparse/sparse_matrix.hpp>
#include <cuda/memory/buffer.hpp>
#include <functional>
#include <vector>

namespace nova {
namespace gnn {

using MessageFunction = std::function<float(float src_feat, float dst_feat)>;
using AggregateFunction = std::function<float(const std::vector<float>& messages)>;

class MessagePassing {
public:
    MessagePassing() = default;

    void set_message_function(MessageFunction fn) { message_fn_ = std::move(fn); }
    void set_aggregate_function(AggregateFunction fn) { aggregate_fn_ = std::move(fn); }

    void forward(const sparse::SparseMatrixCSR<float>& graph,
                 const float* node_features,
                 float* output);

    void gcn_aggregate(const sparse::SparseMatrixCSR<float>& graph,
                       const float* node_features,
                       float* output);

    void gcn_propagate(const sparse::SparseMatrixCSR<float>& graph,
                       const float* node_features,
                       float* output,
                       int hops = 1);

private:
    MessageFunction message_fn_;
    AggregateFunction aggregate_fn_;
};

void MessagePassing::forward(const sparse::SparseMatrixCSR<float>& graph,
                              const float* node_features,
                              float* output) {
    int num_nodes = graph.num_rows();

    for (int i = 0; i < num_nodes; ++i) {
        std::vector<float> messages;

        for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
            int neighbor = graph.col_indices()[idx];
            float msg;

            if (message_fn_) {
                msg = message_fn_(node_features[neighbor], node_features[i]);
            } else {
                msg = node_features[neighbor];
            }

            messages.push_back(msg);
        }

        if (aggregate_fn_ && !messages.empty()) {
            output[i] = aggregate_fn_(messages);
        } else if (!messages.empty()) {
            float sum = 0.0f;
            for (float m : messages) sum += m;
            output[i] = sum;
        } else {
            output[i] = 0.0f;
        }
    }
}

void MessagePassing::gcn_aggregate(const sparse::SparseMatrixCSR<float>& graph,
                                    const float* node_features,
                                    float* output) {
    int num_nodes = graph.num_rows();

    for (int i = 0; i < num_nodes; ++i) {
        float sum = 0.0f;
        int degree = graph.row_offsets()[i + 1] - graph.row_offsets()[i];

        for (int idx = graph.row_offsets()[i]; idx < graph.row_offsets()[i + 1]; ++idx) {
            int neighbor = graph.col_indices()[idx];
            sum += node_features[neighbor];
        }

        output[i] = (degree > 0) ? (sum / std::sqrt(static_cast<float>(degree + 1))) : 0.0f;
    }
}

void MessagePassing::gcn_propagate(const sparse::SparseMatrixCSR<float>& graph,
                                    const float* node_features,
                                    float* output,
                                    int hops) {
    if (hops <= 1) {
        gcn_aggregate(graph, node_features, output);
        return;
    }

    std::vector<float> temp(node_features, node_features + graph.num_rows());
    std::vector<float> next(graph.num_rows());

    gcn_aggregate(graph, temp.data(), next.data());

    for (int h = 2; h <= hops; ++h) {
        gcn_aggregate(graph, next.data(), temp.data());
        std::copy(temp.begin(), temp.end(), next.begin());
    }

    std::copy(next.begin(), next.end(), output);
}

} // namespace gnn
} // namespace nova

#endif // NOVA_CUDA_GNN_MESSAGE_PASSING_HPP
