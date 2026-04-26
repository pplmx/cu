# Nova CUDA Library Enhancement

## What This Is

A production-ready CUDA parallel algorithms library with a five-layer architecture, supporting education, extensibility, and production use cases. This project adds production-quality foundations and new algorithm capabilities.

## Current Milestone: v1.6 Performance & Training

**Previous milestone:** v1.5 Fault Tolerance — SHIPPED 2026-04-26

**Goal:** Enhance training performance with distributed batch normalization, profiling infrastructure, and kernel fusion opportunities.

**Target features:**
- Distributed batch normalization with cross-GPU synchronization
- Performance profiling infrastructure for kernel and collective operations
- Kernel fusion for matmul-bias-activation patterns
- Memory optimization with checkpoint compression and gradient buffering

## Core Value

A reliable, high-performance CUDA compute library that can be trusted in production environments, with comprehensive algorithms for scientific computing, image processing, and emerging workloads.

## Requirements

### Validated

- ✓ Five-layer CUDA architecture (memory → device → algo → api) — existing
- ✓ Memory management (Buffer, unique_ptr, MemoryPool) — existing
- ✓ Algorithm wrappers (reduce, scan, sort, histogram) — existing
- ✓ Image processing (blur, sobel, morphology, brightness) — existing
- ✓ Matrix operations (add, mult, ops) — existing
- ✓ Device capability queries and auto block size selection — v1.0
- ✓ Memory pool statistics and fragmentation reporting — v1.0
- ✓ Stream-based async operations with event synchronization — v1.0
- ✓ Signal/image processing via FFT — v1.0
- ✓ Ray tracing primitives (ray-box, ray-sphere, BVH) — v1.0
- ✓ Graph processing (BFS, PageRank) — v1.0
- ✓ Deep learning primitives (matmul, activations, normalization) — v1.0
- ✓ Device mesh detection and peer memory access — v1.1
- ✓ Multi-GPU data parallelism primitives (reduce, broadcast, all-gather, barrier) — v1.1
- ✓ Distributed memory pool across GPU devices — v1.1
- ✓ Multi-GPU matmul with single-GPU fallback — v1.1
- ✓ C++23 standard (CMAKE_CXX_STANDARD 23) — v1.2
- ✓ CUDA 20 standard (CMAKE_CUDA_STANDARD 20) — v1.2
- ✓ CMake 4.0+ minimum version — v1.2
- ✓ 444 tests passing — v1.2
- ✓ NCCL integration for optimized multi-GPU collectives — v1.3
- ✓ Extended NCCL collectives with unified fallback — v1.3
- ✓ Tensor parallelism for large layer support — v1.3
- ✓ Pipeline parallelism for deep model support — v1.3
- ✓ MPI-based NCCL initialization for inter-node communication — v1.4
- ✓ Topology-aware collective selection across nodes — v1.4
- ✓ Cross-node NCCL communicator management — v1.4

### Active

- [ ] Distributed batch normalization with cross-GPU sync — Phase 25
- [ ] Performance profiling infrastructure — Phase 26
- [ ] Kernel fusion for training efficiency — Phase 27
- [ ] Memory optimization (compression, accumulation) — Phase 28

### Completed (v1.5)

- [x] GPU checkpoint/restart with full state serialization — Phase 21
- [x] Communication error recovery for NCCL/TCP failures — Phase 22
- [x] Memory error detection and ECC error handling — Phase 23
- [x] Job preemption signal handling — Phase 24

### Out of Scope

- Python bindings — separate project
- Real-time video processing pipeline — not in scope

## Context

**Project:** nova CUDA library at `https://github.com/pplmx/nova`
- **Current:** C++23, CUDA 20, CMake 4.0+
- Target architectures: 6.0, 7.0, 8.0, 9.0 (Pascal through Ampere)
- Five-layer architecture with clear separation of concerns
- **444 tests using Google Test v1.14.0**
- **v1.2 shipped:** Toolchain upgrade (C++23, CUDA 20, CMake 4.0)

**Current capabilities:**
- Device mesh detection and peer memory access between GPUs
- Multi-GPU collective operations (all-reduce, broadcast, all-gather, barrier)
- Distributed memory pool spanning multiple GPUs
- Multi-GPU matrix multiply with single-GPU fallback
- All v1.0-v1.5 features: FFT, Ray Tracing, Graph Algorithms, Neural Net Primitives, Async/Streaming, NCCL, Tensor Parallelism, Pipeline Parallelism, Fault Tolerance

**Added in v1.4:**
- MPI-based NCCL bootstrapping for multi-node
- Topology-aware collective algorithm selection
- Hierarchical cross-node communicators

**Added in v1.5:**
- GPU checkpoint/restart with full state serialization
- Communication error recovery with exponential backoff
- Memory error detection and device health monitoring
- Job preemption signal handling (SIGTERM/SIGUSR1)

**Added in v1.6 (planned):**
- NCCL 2.25+ integration with P2P fallback
- Stream-based NCCL collectives with async error handling
- Column/row parallel matmul for transformer layers
- TensorParallelLayer abstractions
- PipelineScheduler with 1F1B and interleaved schedules
- P2P send/recv for inter-stage communication

## Constraints

- **Tech stack:** C++23, CUDA 20, CMake 4.0+ — current versions
- **Backward compatibility:** Existing API must not break
- **Testing:** All existing tests must pass after upgrade
- **Performance:** New implementations must not regress existing algorithms

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Foundation-first phasing | Quality foundations enable reliable feature work | ✓ v1.0 shipped |
| Streams for async | Native CUDA streams, not abstraction layer | ✓ Implemented |
| FFTW-style API | Familiar interface for signal processing users | ✓ Implemented |
| BVH helpers over full ray tracer | Focus on GPU compute primitives | ✓ Implemented |
| P2P ring-allreduce fallback | No NCCL dependency for v1.1 | ✓ Implemented |
| Row-wise split matmul | Simple, builds on existing infrastructure | ✓ Implemented |
| Device mesh singleton | Lazy initialization, single source of truth | ✓ Implemented |
| C++23 adoption | std::expected, constexpr, ranges for modern patterns | ✓ v1.2 shipped |
| CUDA 20 standard | Next-generation CUDA toolkit for new features | ✓ v1.2 shipped |
| CMake 4.0+ | Modern CMake features and policy support | ✓ v1.2 shipped |
| Optional NCCL with P2P fallback | Preserve single-node without NCCL | ✓ v1.3 shipped |
| TensorParallelMatmul (col/row) | Build on existing DistributedMatmul | ✓ v1.3 shipped |
| 1F1B pipeline scheduler | Classic GPipe-style scheduling | ✓ v1.3 shipped |
| MPI for multi-node init | Standard for cluster NCCL bootstrapping | ✓ v1.4 shipped |
| Checkpoint granularity | Full state (weights + optimizer + RNG) | ✓ v1.5 shipped |
| Error recovery strategy | Detect → isolate → recover → retry | ✓ v1.5 shipped |
| Signal handling | SIGTERM/SIGUSR1 for graceful shutdown | ✓ v1.5 shipped |
| Thread-safety | Mutex protection for signal state | ✓ v1.5 shipped |
| BatchNorm strategy | SyncBatchNorm with NCCL all-reduce | v1.6 planning |
| Profiling approach | Integrate with CUDA profiling tools | v1.6 planning |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state (users, feedback, metrics)

---
*Last updated: 2026-04-26 after v1.6 milestone planning*
