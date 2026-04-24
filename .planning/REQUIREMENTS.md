# Requirements: Nova CUDA Library Enhancement

**Defined:** 2026-04-24
**Milestone:** v1.3 NCCL Integration, Tensor & Pipeline Parallelism
**Core Value:** Enable efficient multi-GPU training with NCCL-based collectives, tensor parallelism for large layers, and pipeline parallelism for deep models.

## v1 Requirements

Requirements for v1.3 milestone. Each maps to roadmap phases.

### NCCL Foundation (Phase 1)

- [ ] **NCCL-01**: Library detection and version validation via CMake find module
- [ ] **NCCL-02**: NcclContext with dependency injection pattern and DeviceMesh integration
- [ ] **NCCL-03**: Communicator initialization and lifecycle management per device
- [ ] **NCCL-04**: Shared memory validation (require 512MB+) with clear error messages
- [ ] **NCCL-05**: Async error polling infrastructure with ncclCommGetAsyncError

### Core Collectives (Phase 2)

- [ ] **COLL-01**: Stream-based all-reduce replacing P2P ring-allreduce fallback
- [ ] **COLL-02**: Broadcast wrapper for weight synchronization across devices
- [ ] **COLL-03**: Barrier implementation for explicit synchronization points
- [ ] **COLL-04**: Safe NCCL call wrapper with async error detection
- [ ] **COLL-05**: Stream-ordered collectives passing cudaStream_t to all operations

### Extended Collectives (Phase 3)

- [ ] **EXTD-01**: All-gather for row-parallel activation gathering
- [ ] **EXTD-02**: Reduce-scatter for alternative gradient aggregation
- [ ] **EXTD-03**: Group operations with ncclGroupStart/End batching
- [ ] **EXTD-04**: Unified NCCL/legacy fallback path for deployment flexibility
- [ ] **EXTD-05**: Communicator caching for repeated collective operations

### Tensor Parallelism (Phase 4)

- [ ] **TENS-01**: TensorParallelMatmul with column-parallel strategy
- [ ] **TENS-02**: TensorParallelMatmul with row-parallel strategy
- [ ] **TENS-03**: ColumnParallelLayer for QKV projection pattern
- [ ] **TENS-04**: RowParallelLayer for output projection pattern
- [ ] **TENS-05**: Integration with existing DistributedMatmul infrastructure
- [ ] **TENS-06**: Memory-aware TP degree selection with profiling

### Pipeline Parallelism (Phase 5)

- [ ] **PIPE-01**: PipelineScheduler with 1F1B schedule implementation
- [ ] **PIPE-02**: P2P send/recv primitives for inter-stage communication
- [ ] **PIPE-03**: Activation buffer management with ping-pong overlap
- [ ] **PIPE-04**: Communicator splitting via ncclCommSplit for TP+DP
- [ ] **PIPE-05**: Interleaved schedule option for reduced bubble overhead
- [ ] **PIPE-06**: Stage balance validation within 10% compute variance

## v2 Requirements

Deferred to future release.

### Distributed Batch Normalization

- **DBN-01**: Cross-GPU batch statistics aggregation
- **DBN-02**: Distributed sync BatchNorm layer
- **DBN-03**: Memory-efficient distributed BatchNorm with tensor parallelism

### Multi-Node Support

- **MULN-01**: MPI-based NCCL initialization for inter-node communication
- **MULN-02**: Topology-aware collective selection across nodes
- **MULN-03**: NCCL communicator across multiple nodes

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Hard NCCL dependency | Must preserve P2P fallback for environments without NCCL |
| DeepSpeed integration | Over-engineered for single-node scope |
| Megatron-LM integration | Too opinionated, single-node simplicity preferred |
| FlashAttention integration | Sequence parallelism is separate concern |
| NVSHMEM | Single-node scope, InfiniBand not required |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| NCCL-01 | Phase 1 | Pending |
| NCCL-02 | Phase 1 | Pending |
| NCCL-03 | Phase 1 | Pending |
| NCCL-04 | Phase 1 | Pending |
| NCCL-05 | Phase 1 | Pending |
| COLL-01 | Phase 2 | Pending |
| COLL-02 | Phase 2 | Pending |
| COLL-03 | Phase 2 | Pending |
| COLL-04 | Phase 2 | Pending |
| COLL-05 | Phase 2 | Pending |
| EXTD-01 | Phase 3 | Pending |
| EXTD-02 | Phase 3 | Pending |
| EXTD-03 | Phase 3 | Pending |
| EXTD-04 | Phase 3 | Pending |
| EXTD-05 | Phase 3 | Pending |
| TENS-01 | Phase 4 | Pending |
| TENS-02 | Phase 4 | Pending |
| TENS-03 | Phase 4 | Pending |
| TENS-04 | Phase 4 | Pending |
| TENS-05 | Phase 4 | Pending |
| TENS-06 | Phase 4 | Pending |
| PIPE-01 | Phase 5 | Pending |
| PIPE-02 | Phase 5 | Pending |
| PIPE-03 | Phase 5 | Pending |
| PIPE-04 | Phase 5 | Pending |
| PIPE-05 | Phase 5 | Pending |
| PIPE-06 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 26
- Unmapped: 0

---
*Requirements defined: 2026-04-24*
*Last updated: 2026-04-24 after v1.3 requirements definition*
