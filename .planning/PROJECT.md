# Nova CUDA Library Enhancement

## What This Is

A production-ready CUDA parallel algorithms library with a five-layer architecture, supporting education, extensibility, and production use cases. This project adds production-quality foundations and new algorithm capabilities.

## Current Milestone: v1.1 Multi-GPU Support

**Goal:** Enable distributed GPU compute across device meshes with data parallelism primitives and multi-GPU matmul.

**Target features:**
- Device mesh detection and peer memory access
- Multi-GPU data parallelism primitives (reduce, broadcast, all-gather, distributed batch norm)
- Distributed memory pool across GPU devices
- Multi-GPU matmul with approach TBD (tensor parallelism, pipeline parallelism, or hybrid)

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
- ✓ 81+ tests across 13 test suites — existing
- ✓ CMake build with Google Test — existing

### Active

- [ ] Device mesh detection and peer memory access (MGPU-01)
- [ ] Multi-GPU data parallelism primitives (MGPU-02)
- [ ] Distributed memory pool across GPU devices (MGPU-03)
- [ ] Multi-GPU matmul (MGPU-04)

### Out of Scope

- Distributed multi-node computation (multiple nodes, not just multiple GPUs) — future work
- Python bindings — separate project
- Real-time video processing pipeline — not in scope

## Context

**Project:** nova CUDA library at `https://github.com/pplmx/nova`
- C++20, CUDA 17, CMake 3.25+
- Target architectures: 6.0, 7.0, 8.0, 9.0 (Pascal through Ampere)
- Five-layer architecture with clear separation of concerns
- 81+ tests using Google Test v1.14.0
- v1.0 shipped: FFT, Ray Tracing, Graph Algorithms, Neural Net Primitives, Async/Streaming
- **v1.1 focus:** Multi-GPU support (device mesh, data parallelism, distributed memory pool, multi-GPU matmul)

**Known limitations from codebase map:**
- No multi-GPU support (v1.1 target)
- No peer memory access between devices
- No distributed memory pool across GPUs

## Constraints

- **Tech stack:** C++20, CUDA 17, CMake 3.25+ — must maintain compatibility
- **Backward compatibility:** Existing API must not break
- **Testing:** All new code requires tests, maintain 80%+ coverage
- **Performance:** New implementations must not regress existing algorithms

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Foundation-first phasing | Quality foundations enable reliable feature work | ✓ v1.0 shipped |
| Streams for async | Native CUDA streams, not abstraction layer | ✓ Implemented |
| FFTW-style API | Familiar interface for signal processing users | ✓ Implemented |
| BVH helpers over full ray tracer | Focus on GPU compute primitives | ✓ Implemented |
| CUDA MPS for multi-GPU shared memory | Native, low-overhead GPU sharing | — TBD |
| Multi-GPU matmul approach | Tensor vs pipeline vs hybrid parallelism | — Research in v1.1 |

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
*Last updated: 2026-04-24 — Milestone v1.1 started*
