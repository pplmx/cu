# Milestones

## v1.0 Production Release (Shipped: 2026-04-24)

**Phases completed:** 6 phases, 9 plans, 58 requirements

**Key accomplishments:**

- **Phase 1: Performance Foundations** - Device-aware kernels, memory metrics, validation framework, and comprehensive benchmark suite
- **Phase 2: Async & Streaming** - CUDA stream manager with priorities, pinned memory allocator, async copy primitives, and memory pool v2 with defragmentation
- **Phase 3: FFT Module** - Fast Fourier Transform implementation with forward/inverse transforms and plan management
- **Phase 4: Ray Tracing Primitives** - GPU-accelerated ray intersection (box, sphere) and BVH construction with 29 tests
- **Phase 5: Graph Algorithms** - BFS and PageRank on GPU using CSR storage format
- **Phase 6: Neural Net Primitives** - Matrix multiply, softmax, leaky ReLU, and layer normalization kernels

**Requirements delivered:** 58 total (PERF-01 to PERF-06, BMCH-01 to BMCH-04, ASYNC-01 to ASYNC-04, POOL-01 to POOL-04, FFT-01 to FFT-04, RAY-01 to RAY-04, GRAPH-01 to GRAPH-04, NN-01 to NN-04)

**Core features implemented:**
- Device capability queries and auto block size selection
- Memory pool statistics and fragmentation reporting
- Stream-based async operations with event synchronization
- Signal/image processing via FFT
- Scientific computing primitives for ray tracing
- Graph processing (traversal, ranking)
- Deep learning primitives (matmul, activations, normalization)

---

## v1.1 Multi-GPU Support (Shipped: 2026-04-24)

**Phases completed:** 4 phases, 4 plans, 13 requirements

**Key accomplishments:**

- **Phase 7: Device Mesh Detection** - DeviceMesh, PeerCapabilityMap, ScopedDevice, PeerCopy with 25 tests passing
- **Phase 8: Multi-GPU Data Parallelism** - DistributedReduce, DistributedBroadcast, DistributedAllGather, MeshBarrier primitives
- **Phase 9: Distributed Memory Pool** - Per-device pools with auto-allocation, ownership tracking, cross-device visibility
- **Phase 10: Multi-GPU Matmul** - Row-wise split with single-GPU fallback, 11 tests passing

**Requirements delivered:** 13 total (MGPU-01 to MGPU-13)

**Core features implemented:**
- Device mesh detection and peer memory access between GPUs
- Multi-GPU collective operations (all-reduce, broadcast, all-gather, barrier)
- Distributed memory pool spanning multiple GPUs
- Multi-GPU matrix multiply with single-GPU fallback

**Next:** v1.2 Toolchain Upgrade (C++23, CUDA 20, CMake 4.0+)

---

## v1.2 Toolchain Upgrade (Shipped: 2026-04-24)

**Phases completed:** 2 phases, 2 plans, 9 requirements

**Key accomplishments:**

- **Phase 11: Toolchain Analysis** - Compatibility audit for C++23, CUDA 20, CMake 4.0+
- **Phase 12: Toolchain Upgrade** - CMakeLists.txt updates, 444 tests passing

**Requirements delivered:** 9 total (TC-01 to TC-09)

**Core features implemented:**
- C++23 standard (CMAKE_CXX_STANDARD 23)
- CUDA 20 standard (CMAKE_CUDA_STANDARD 20)
- CMake 4.0+ minimum version
- All 444 tests passing

**Future roadmap:** v1.3 with NCCL integration, tensor parallelism, and pipeline parallelism
