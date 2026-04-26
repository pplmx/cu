# Requirements — v2.1 New Algorithms

## Active Requirements

### Sparse Matrix Support

- [ ] **SPARSE-01**: User can create sparse matrices in CSR format
- [ ] **SPARSE-02**: User can create sparse matrices in CSC format
- [ ] **SPARSE-03**: User can perform sparse matrix-dense vector multiplication (SpMV)
- [ ] **SPARSE-04**: User can perform sparse matrix-dense matrix multiplication (SpMM)

### Graph Neural Networks

- [ ] **GNN-01**: User can run message passing on GPU graphs
- [ ] **GNN-02**: User can compute graph attention mechanisms
- [ ] **GNN-03**: User can run graph sampling operations
- [ ] **GNN-04**: User can execute multi-hop neighborhood aggregation

### Quantization

- [ ] **QUANT-01**: User can quantize tensors to INT8 format
- [ ] **QUANT-02**: User can quantize tensors to FP16 format
- [ ] **QUANT-03**: User can perform quantized matmul operations
- [ ] **QUANT-04**: User can use mixed precision (FP32/FP16/INT8) in same computation

## Future Requirements (Deferred)

- Sparse matrix COO format support
- Graph pooling operations
- QAT (Quantization-Aware Training) support

## Out of Scope

- Dense graph algorithms — already implemented
- CPU-only operations — GPU focused
- Third-party framework integration (PyTorch, TensorFlow)

## Traceability

| Phase | REQ-ID | Description | Status |
|-------|--------|-------------|--------|
| TBD | SPARSE-01 | CSR format creation | TBD |
| TBD | SPARSE-02 | CSC format creation | TBD |
| TBD | SPARSE-03 | SpMV operation | TBD |
| TBD | SPARSE-04 | SpMM operation | TBD |
| TBD | GNN-01 | Message passing | TBD |
| TBD | GNN-02 | Graph attention | TBD |
| TBD | GNN-03 | Graph sampling | TBD |
| TBD | GNN-04 | Neighborhood aggregation | TBD |
| TBD | QUANT-01 | INT8 quantization | TBD |
| TBD | QUANT-02 | FP16 quantization | TBD |
| TBD | QUANT-03 | Quantized matmul | TBD |
| TBD | QUANT-04 | Mixed precision | TBD |

---
*Requirements defined: 2026-04-26 for v2.1 New Algorithms*
