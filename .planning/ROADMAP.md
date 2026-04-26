# Roadmap — v2.1 New Algorithms

## Milestone Summary

| Metric | Value |
|--------|-------|
| Milestone | v2.1 New Algorithms |
| Requirements | 12 |
| Phases | 4 |
| Started | 2026-04-26 |

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 44 | Sparse Matrix Support | Add CSR/CSC formats and sparse operations (SpMV, SpMM) | SPARSE-01, SPARSE-02, SPARSE-03, SPARSE-04 | 4 criteria |
| 45 | Graph Neural Networks | Implement GNN primitives for message passing and attention | GNN-01, GNN-02, GNN-03, GNN-04 | 4 criteria |
| 46 | Quantization Foundation | Build quantization infrastructure (INT8, FP16) | QUANT-01, QUANT-02 | 2 criteria |
| 47 | Quantized Operations | Implement quantized matmul and mixed precision | QUANT-03, QUANT-04 | 2 criteria |

---

## Phase 44: Sparse Matrix Support

**Goal:** Add CSR/CSC sparse matrix formats and sparse operations (SpMV, SpMM)

**Requirements:** SPARSE-01, SPARSE-02, SPARSE-03, SPARSE-04

**Success Criteria:**
1. User can create `SparseMatrixCSR` from dense input with `SparseMatrixCSR::FromDense()`
2. User can create `SparseMatrixCSC` from CSR with `SparseMatrixCSR::ToCSC()`
3. User can run SpMV: `sparse_mv(matrix, vector, result)` produces correct output
4. User can run SpMM: `sparse_mm(matrix_a, dense_b, result)` produces correct output

**Dependencies:** None (foundation phase)

**Key Decisions:**
- Use cuSPARSE for efficient sparse operations where available
- Provide fallback implementations for non-CUDA builds
- CSR as primary format, CSC for column-wise operations

---

## Phase 45: Graph Neural Networks

**Goal:** Implement GNN primitives for message passing and graph attention

**Requirements:** GNN-01, GNN-02, GNN-03, GNN-04

**Success Criteria:**
1. User can run `MessagePassing(graph, features, message_fn, aggregate_fn)` with GPU acceleration
2. User can compute `GraphAttention(node_features, edge_index, heads)` with attention weights
3. User can sample `GraphSampler.sample_neighbors()` for mini-batch training
4. User can execute `k_hop_aggregation()` for multi-hop neighborhood features

**Dependencies:** Phase 44 (uses CSR graph format)

**Key Decisions:**
- Build on existing CSR graph infrastructure
- Support both GCN and GAT aggregation styles
- Sampler operates on CSR format for efficiency

---

## Phase 46: Quantization Foundation

**Goal:** Build quantization infrastructure for INT8 and FP16 tensors

**Requirements:** QUANT-01, QUANT-02

**Success Criteria:**
1. User can quantize `QuantizedTensor<int8_t>::FromFloat(float_tensor, scale)` with calibration
2. User can quantize `QuantizedTensor<float16_t>::FromFloat(float_tensor)` with type conversion
3. User can dequantize back to float32 for verification
4. User can inspect quantization metadata (scale, zero_point, dtype)

**Dependencies:** Phase 44 (uses tensor infrastructure)

**Key Decisions:**
- Support per-tensor and per-channel quantization
- Calibration-based quantization (not QAT initially)
- Store quantization metadata alongside tensor data

---

## Phase 47: Quantized Operations

**Goal:** Implement quantized matmul and mixed precision computation

**Requirements:** QUANT-03, QUANT-04

**Success Criteria:**
1. User can run `quantized_matmul(q_a, q_b, result)` producing quantized output
2. User can execute `mixed_precision_matmul(fp32_a, int8_b, fp16_c)` with automatic casting
3. Quantized operations maintain accuracy within 1% of FP32 baseline
4. User can toggle between quantized and FP32 paths at runtime

**Dependencies:** Phase 46 (quantization infrastructure required)

**Key Decisions:**
- Use WMMA (Warp Matrix Multiply and Accumulate) for INT8 on Tensor Cores
- Automatic precision promotion in mixed operations
- Runtime dispatch to fastest available implementation

---

## Requirement Coverage Matrix

| REQ-ID | Phase | Covered |
|--------|-------|---------|
| SPARSE-01 | 44 | ✅ |
| SPARSE-02 | 44 | ✅ |
| SPARSE-03 | 44 | ✅ |
| SPARSE-04 | 44 | ✅ |
| GNN-01 | 45 | ✅ |
| GNN-02 | 45 | ✅ |
| GNN-03 | 45 | ✅ |
| GNN-04 | 45 | ✅ |
| QUANT-01 | 46 | ✅ |
| QUANT-02 | 46 | ✅ |
| QUANT-03 | 47 | ✅ |
| QUANT-04 | 47 | ✅ |

**All 12 requirements mapped across 4 phases.**

---

## Execution Order

```
Phase 44 (Sparse Matrix Support) ← Foundation
         ↓
Phase 45 (Graph Neural Networks) ← Builds on sparse/CSR
         ↓
Phase 46 (Quantization Foundation) ← Independent, can parallel
         ↓
Phase 47 (Quantized Operations) ← Requires Phase 46
```

---
*Roadmap created: 2026-04-26 for v2.1 New Algorithms*
*4 phases, 12 requirements, all covered*
