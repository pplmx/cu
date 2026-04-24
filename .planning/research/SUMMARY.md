# Research Summary: Nova CUDA Library v1.3

**Project:** NCCL Integration, Tensor Parallelism, Pipeline Parallelism
**Date:** 2026-04-24
**Overall Confidence:** HIGH (NCCL docs), MEDIUM (TP/Pipeline patterns)

---

## Executive Summary

Nova v1.3 adds production-grade NCCL collective operations and distributed parallelism strategies for large model support. The core insight: NCCL provides hardware-aware communication (ring, tree, collnet) that automatically adapts to topology—a strict superset of existing P2P fallbacks. Key decisions: (1) NCCL 2.25+ with optional fallback, (2) custom tensor parallelism (avoid Megatron-LM/DeepSpeed complexity), (3) GPipe-style pipeline scheduling with 1F1B overlap.

The most critical engineering challenge is async error handling. Unlike CUDA API calls that fail immediately, NCCL collective errors surface asynchronously via `ncclCommGetAsyncError()` and can corrupt communicators permanently if not handled defensively. This requires polling-based health checks instead of blocking waits on every collective operation.

---

## Key Findings

### From STACK.md

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **NCCL** | 2.25+ | CUDA 20 support, `ncclCommInitRankConfig`, `ncclCommSplit`, async error handling |
| **Tensor Parallelism** | Custom implementation | Megatron-LM too opinionated; single-node scope simplifies communication |
| **Pipeline Parallelism** | GPipe-style with async micro-batching | Simpler than PipeDream, effective throughput via micro-batching |
| **Avoid** | DeepSpeed, Megatron-LM integration | Over-engineered for single-node, large dependencies |

**Key APIs:** `ncclCommInitRank`, `ncclCommSplit`, `ncclGroupStart/End`, `ncclCommGetAsyncError`

### From FEATURES.md

**Table Stakes (implement first):**
- NCCL communicator initialization via DeviceMesh
- All-reduce, broadcast, barrier wrappers
- Stream-based async collectives
- Error handling with async error polling

**Differentiators (implement second):**
- Unified NCCL/CUDA fallback path
- Automatic device mesh configuration derivation
- Communicator caching
- NVTX profiling integration
- Multi-communicator support for TP+DP parallelism

**Anti-features to never add:**
- Hard NCCL dependency (always provide fallback)
- Blocking operations without streams
- Single-rank collective calls

### From ARCHITECTURE.md

**Critical architectural decisions:**

1. **Dependency injection over singleton for NcclContext** — Enables testing and explicit dependencies; provide `NcclContext::instance()` for convenience
2. **Async collectives with explicit streams** — Pass `cudaStream_t` to all collective calls; never use `cudaStreamNull` in production
3. **Extend DistributedMatmul with TP strategy** — Column-parallel for QKV projection, row-parallel for output projection
4. **PipelineScheduler separates scheduling from computation** — Scheduling logic decoupled from layer execution

**New directory structure:**
```
include/cuda/
    nccl/                    # NEW: NCCL integration
    tensor_parallel/         # NEW: TP implementation  
    pipeline/                # NEW: Pipeline parallelism
    distributed/             # EXTENDED: Add NCCL backend
```

### From PITFALLS.md

**Top 5 pitfalls with prevention:**

| # | Pitfall | Prevention |
|---|---------|------------|
| 1 | **NCCL initialization failures** | Check `/dev/shm` size (require 512MB+), validate NCCL version, Docker: `--shm-size=1g` |
| 2 | **TP memory explosions** | Profile memory per TP degree; replicated optimizer states = 2x model per GPU |
| 3 | **Pipeline bubble overhead** | Balance stage compute within 10%; M >= 4*K microbatches for <10% bubbles |
| 4 | **Cross-collective deadlocks** | Single-threaded dispatch or `NCCL_LAUNCH_ORDER_IMPLICIT=1` |
| 5 | **NCCL timeout hangs** | Never use bare `cudaStreamSynchronize`; poll `ncclCommGetAsyncError()` |

**Integration gotchas:**
- cuFFT/cuBLAS version must match NCCL build
- Default stream (`cudaStreamLegacy`) serializes incorrectly with NCCL
- P2P and NCCL cannot coexist on same peer pair
- CUDA Graphs + multiple communicators deadlock

---

## Implications for Roadmap

### Recommended Phase Structure

**Phase 1: Foundation (NCCL Integration Basics)**
- NCCL library detection and CMake integration
- `NcclContext` with dependency injection pattern
- Basic communicator initialization via DeviceMesh
- **Pitfalls to avoid:** Shared memory exhaustion, version mismatch

**Phase 2: Core Collectives (AllReduce, Broadcast, Barrier)**
- Stream-based all-reduce replacing ring-allreduce
- Broadcast wrapper for weight synchronization
- Barrier implementation
- Async error polling infrastructure (`safe_nccl_call`)
- **Pitfalls to avoid:** Cross-collective deadlocks, timeout hangs

**Phase 3: Extended Collectives (AllGather, ReduceScatter, Group Ops)**
- All-gather for row-parallel activation gathering
- Reduce-scatter for alternative gradient aggregation
- Group operations (`ncclGroupStart/End`) for batching
- Unified NCCL/legacy fallback path
- **Pitfalls to avoid:** Single-rank collective calls, blocking operations

**Phase 4: Tensor Parallelism**
- `TensorParallelMatmul` with column/row parallel strategies
- `ColumnParallelLayer` (QKV projection pattern)
- `RowParallelLayer` (output projection pattern)
- Integration with existing `DistributedMatmul`
- **Pitfalls to avoid:** Memory explosions from replicated optimizer states

**Phase 5: Pipeline Parallelism**
- `PipelineScheduler` with 1F1B and interleaved schedules
- P2P send/recv primitives for inter-stage communication
- Activation buffer management with ping-pong overlap
- Communicator splitting via `ncclCommSplit`
- **Pitfalls to avoid:** Unbalanced stage compute, bubble overhead

### Research Flags

| Phase | Needs Deeper Research | Notes |
|-------|----------------------|-------|
| Phase 1 | NO | NCCL docs provide definitive patterns |
| Phase 2 | NO | Error handling patterns well-documented |
| Phase 3 | NO | Collective patterns standardized |
| Phase 4 | MEDIUM | Memory profiling tooling needed |
| Phase 5 | YES | Interleaved scheduling has implementation nuances |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | Based on NVIDIA NCCL 2.30 documentation |
| Features | **HIGH** | Direct API mapping, established ML patterns |
| Architecture | **HIGH** | Megatron-LM patterns stable, well-documented |
| Pitfalls | **HIGH** | Primary source is NVIDIA NCCL docs |

**Gaps to Address:**
- Memory profiling infrastructure not yet built (needed for Phase 4)
- P2P communication patterns for pipeline less standardized than collectives
- Multi-communicator management edge cases with CUDA Graphs

---

## Sources

- **NCCL 2.30 API:** NVIDIA NCCL Documentation (docs.nvidia.com/deeplearning/nccl)
- **Tensor Parallelism:** Megatron-LM column/row parallel patterns, Transformer Engine
- **Pipeline Parallelism:** GPipe, PipeDream papers; Megatron Core schedules
- **Pitfalls:** NCCL Troubleshooting, GitHub issues #2117, #2119, #2106
