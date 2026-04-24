# Project Research Summary

**Project:** Nova CUDA Library Enhancement — v1.1 Multi-GPU Support
**Domain:** Single-node multi-GPU CUDA compute library
**Researched:** 2026-04-24
**Confidence:** HIGH

## Executive Summary

Multi-GPU support for the Nova CUDA library builds on three foundational tiers, each with increasing dependency complexity. The core insight from research: no new hard dependencies are required for foundational multi-GPU support. CUDA's native Peer Memory Access APIs (since CUDA 6.0) and Stream-Ordered Allocator with IPC pools (since CUDA 11.2) cover device mesh detection, peer memory access, and distributed memory pooling without any external libraries. NCCL is valuable for optimized collectives but is optional for v1.1 — a CUDA-native P2P fallback works for 2-4 GPU configurations.

The implementation follows a strict layer ordering: device mesh → peer memory → collectives → distributed matmul. Each layer enables the next, and no layer requires features from later phases. The design is additive — all existing single-GPU code remains unchanged, and all multi-GPU operations integrate with the existing `cuda::async::StreamManager` without modifications.

Critical pitfalls to address in each phase: peer access validation (Phase 1), cross-GPU synchronization deadlocks (Phase 2), distributed memory pool coherency (Phase 3), and tensor parallelism communication patterns (Phase 4). Single-GPU fallback testing is a cross-cutting concern tested throughout all phases.

## Key Findings

### Recommended Stack

**No new hard dependencies for v1.1.** CUDA-native APIs cover all foundational requirements.

**Core technologies (no new deps):**
- `cudaDeviceCanAccessPeer()` — peer capability queries (CUDA 6.0+)
- `cudaMemcpyAsync()` peer variant — direct GPU-to-GPU copy
- `cudaMemPool_t` with `cudaMemPoolSetAccess` — cross-device memory pool (CUDA 11.2+)
- P2P ring-allreduce as fallback for collectives (no NCCL required)

**Optional (v1.2+):**
- NCCL 2.25+ — optimized collectives for 4+ GPUs, NVLink-aware communication

**Explicitly excluded for single-node scope:**
- NVSHMEM — targets InfiniBand/RDMA fabric, overkill for CUDA IPC
- CUDA MPS — system-level daemon for multi-process workloads, not library-level
- GDRCopy — GPU-to-network RDMA, requires specific hardware

### Expected Features

**Must have (table stakes — v1.1 deliverable):**
- GPU enumeration and device mesh detection — users need to know what hardware exists
- Peer access capability queries — foundation for all peer memory operations
- Async peer-to-peer copy primitives — core data movement primitive
- Multi-GPU reduce (all-reduce, reduce-scatter) — essential for gradient synchronization
- Multi-GPU broadcast — required for weight synchronization
- Multi-GPU all-gather — required for data parallel patterns
- Distributed memory pool across GPUs — memory management for multi-device workloads
- Multi-GPU matmul (row-wise split) — core v1.1 deliverable

**Should have (v1.1 scope):**
- Device mesh topology representation — enables optimal collective algorithm selection
- Synchronization barriers — required for correctness in multi-GPU programs
- Single-GPU fallback — all primitives must work on single-GPU systems

**Defer (v1.2+):**
- Device mesh topology optimization — NVLink-aware scheduling
- Tensor parallelism for very large layers — requires significant API design work
- Pipeline parallelism — complex coordination, secondary priority
- Distributed batch normalization — useful for deep learning, not core compute
- NCCL as primary collective backend — implement DIY fallback first

### Architecture Approach

The existing five-layer architecture extends to a six-layer model with one new layer inserted at 2.5:

```
Layer 0: Device Abstraction (EXTENDED) — DeviceMesh, PeerCapabilityMap
Layer 1: Memory Foundation (EXTENDED) — PeerAllocator, DistributedMemoryPool
Layer 2: Device Kernels — unchanged
Layer 2.5: Peer Transport (NEW) — PeerCopy, MeshBarrier
Layer 3: Algorithm Wrappers — unchanged
Layer 4: Distributed Operations (NEW) — DistributedReduce, DistributedBroadcast, DistributedMatmul
Layer 5: High-Level Distributed API — cuda::distributed::reduce(), cuda::distributed::matmul()
```

**New directories:**
- `include/cuda/mesh/` — DeviceMesh, PeerCopy, MeshBarrier
- `include/cuda/memory/` — PeerAllocator, DistributedMemoryPool (extends existing)
- `include/cuda/distributed/` — DistributedReduce, DistributedBroadcast, DistributedMatmul

**New CMake targets:** `cuda_mesh` (interface), `cuda_multigpu` (static library).

**Backward compatibility:** Additive design. No existing headers modified. All multi-GPU types in new namespaces. `cuda::distributed::*` is opt-in.

### Critical Pitfalls

1. **Peer access without validation** — Always check `cudaDeviceCanAccessPeer()` before enabling. Crash on single-GPU systems and unsupported PCIe topologies.
2. **Memory consistency across GPUs** — Stream dependencies are NOT sufficient for cross-GPU consistency. Use explicit events.
3. **Synchronization deadlocks** — Circular dependencies cause hangs. Design as DAGs.
4. **Single-GPU fallback broken** — Test on single-GPU CI runners. Design single-GPU fast path.
5. **Tensor parallelism communication** — Column-parallel needs all-reduce after matmul. Row-parallel needs all-gather before. Easy to get backwards.
6. **NCCL initialization races** — Mutex-protected singleton pattern, destroy before CUDA context cleanup.
7. **Distributed memory pool coherency** — Track which device owns which memory. Prevent double-free.
8. **Stream per device scope** — Streams are device-local. Maintain one stream per device.

## Implications for Roadmap

Based on research, four phases are recommended:

### Phase 1: Device Mesh Detection (MGPU-01)
**Rationale:** Foundation for all multi-GPU work. Must be implemented before any peer communication.
**Delivers:** `DeviceMesh` singleton, `PeerCapabilityMap`, `ScopedDevice` RAII guard, async peer copy primitive.
**Addresses:** PITFALL-1 (peer validation), PITFALL-6 (single-GPU fallback), PITFALL-8 (stream per device).
**Builds on:** Existing `cuda::performance::DeviceInfo`.

### Phase 2: Multi-GPU Data Parallelism Primitives (MGPU-02)
**Rationale:** Collectives are required for multi-GPU matmul and gradient synchronization. Build P2P fallback first, add NCCL in v1.2.
**Delivers:** `DistributedReduce` (ring all-reduce), `DistributedBroadcast`, `DistributedAllGather`, `MeshBarrier`.
**Addresses:** PITFALL-3 (deadlocks), PITFALL-5 (NCCL races).
**Uses:** `PeerCopy` from Phase 1.

### Phase 3: Distributed Memory Pool (MGPU-03)
**Rationale:** Memory management must support multi-device workloads. Extends existing pool pattern.
**Delivers:** `DistributedMemoryPool`, per-device pool coordination, cross-device allocation tracking.
**Addresses:** PITFALL-9 (pool coherency).
**Uses:** Phase 1 device mesh.

### Phase 4: Multi-GPU Matmul (MGPU-04)
**Rationale:** Core v1.1 deliverable. Start with row-wise split (simplest).
**Delivers:** `DistributedMatmul` with `ParallelismStrategy` enum, single-GPU fallback, row-partitioned matmul.
**Addresses:** PITFALL-7 (tensor parallelism patterns).
**Uses:** All prior phases.

### Phase Ordering Rationale

- Phase 1 must precede all others — device mesh is prerequisite to peer communication.
- Phase 2 and 3 can proceed in parallel after Phase 1 — they use the same peer access foundation but don't depend on each other.
- Phase 4 requires Phases 1, 2, and 3 — needs device mesh, collectives, and memory pool.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 2:** NCCL API surface design — which NCCL calls to wrap, how to expose both P2P fallback and NCCL backends
- **Phase 4:** Row-wise vs column-wise split tradeoffs — which to implement first depends on target workload mix

**Phases with standard patterns (skip research-phase):**
- **Phase 1:** CUDA peer APIs are well-documented with clear error codes
- **Phase 3:** Pool pattern already exists in codebase — extend, don't invent

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | CUDA native APIs are stable (CUDA 6.0-17+). NCCL exclusion validated. |
| Features | HIGH | All features map to established CUDA/NCCL APIs. |
| Architecture | HIGH | Additive design, existing patterns extended. |
| Pitfalls | HIGH | Documented with prevention strategies. |

**Overall confidence:** HIGH

### Gaps to Address

- **Multi-GPU matmul strategy:** Row-wise split confirmed as starting point. Tensor parallelism deferred to v1.2+.
- **NCCL API choice:** v1.1 uses P2P fallback. NCCL integration in v1.2 needs decision on NCCL C API vs. wrapped communicator.
- **Topology optimization:** Phase 1 covers basic topology detection. NVLink-aware scheduling is v1.2.

## Sources

### Primary (HIGH confidence)
- NVIDIA CUDA C++ Programming Guide, Sections 6.2.9 (Multi-Device), 15 (Memory Allocator), 15.10-15.11 (IPC Pools)
- NCCL 2.30 Documentation: Collectives, API Reference
- Existing Nova codebase: `MemoryPool`, `StreamManager`, `DeviceInfo`, `Matmul` patterns

### Secondary (MEDIUM confidence)
- Community patterns for ring-allreduce P2P fallback
- Phase-ordering from feature dependency analysis

---

*Research completed: 2026-04-24*
*Ready for roadmap: yes*
