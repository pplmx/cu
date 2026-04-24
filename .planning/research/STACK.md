# Technology Stack: Multi-GPU Support for Nova CUDA Library

**Project:** Nova CUDA Library v1.1
**Researched:** 2026-04-24
**Confidence:** HIGH (CUDA built-in), MEDIUM (NCCL ecosystem), LOW (NVSHMEM single-node — needs validation)

## Recommendation Summary

Build multi-GPU support in three tiers, adding dependencies only where necessary:

| Tier | Technology | Dependency? | When to Use |
|------|-----------|-------------|-------------|
| 1 - Foundational | CUDA Peer Memory Access APIs | None (CUDA built-in) | MGPU-01: Device mesh, peer access |
| 2 - Memory Pool | CUDA Stream-Ordered Allocator + IPC Pools | None (CUDA 11.2+) | MGPU-03: Distributed memory pool |
| 3 - Collectives | NCCL 2.25+ | **Optional**, CNMEM-bundled | MGPU-02, MGPU-04: only if collectves needed |

**Do NOT add NVSHMEM for single-node.** NVSHMEM targets GPU-initiated communication and InfiniBand/nvlink fabric. For single-node multi-GPU with CUDA IPC transport, it adds unnecessary complexity and external hardware dependencies. Revisit only if multi-node becomes a future requirement.

## Technology Analysis

### Tier 1: CUDA Peer Memory Access (No New Dependencies)

**CUDA Version Requirement:** CUDA 6.0+ (cudaDeviceCanAccessPeer), fully compatible with CUDA 17.

These are native CUDA Runtime APIs, no extra libraries or daemons needed.

#### APIs to Add

| API | Purpose | Docs |
|-----|---------|------|
| `cudaDeviceCanAccessPeer()` | Query if two devices can P2P access | Programming Guide 6.2.9.4 |
| `cudaEnablePeerAccess()` | Enable P2P memory access between devices | Programming Guide 6.2.9.4 |
| `cudaDisablePeerAccess()` | Disable P2P access | Programming Guide 6.2.9.4 |
| `cudaMemcpyAsync()` peer variant | Direct GPU-to-GPU copy | Programming Guide 6.2.9.5 |
| `cudaIpcGetMemHandle()` | Export memory for cross-process sharing | Programming Guide 15.11 |
| `cudaIpcOpenMemHandle()` | Import shared memory handle | Programming Guide 15.11 |

#### Integration with Existing Layers

Fits cleanly into the existing `device` layer as a new `DeviceMesh` class:

```cpp
// New: include/cuda/device/device_mesh.h
// Fits existing namespace: cuda::device

class DeviceMesh {
public:
    struct PeerInfo {
        int device_id;
        bool can_access_peer;
        int access_direction;  // 0=none, 1=to, 2=from, 3=bidirectional
    };

    // Detects all GPUs and builds peer access graph
    static std::vector<PeerInfo> detect_mesh();
    
    // Enable P2P access for all mutually-accessible device pairs
    static void enable_peer_access();
    
    // Check if two specific devices can P2P
    static bool can_access(int src_device, int dst_device);
};
```

**Integration point with existing stream layer:** The existing `cuda::async::StreamManager` can spawn per-device streams. After enabling peer access, the existing `cudaStream_t` handles can be used for `cudaMemcpyAsync` peer copies — no changes to the stream abstraction needed.

**Existing error handling:** Extend `cuda/device/error.h` with a `CUDA_PEER_CHECK` macro mirroring the existing `CUDA_CHECK`.

#### What This Enables

- MGPU-01: Device mesh detection and peer memory access
- MGPU-03: Peer-aware memory pool allocation (allocate on device with most free memory)
- The foundation for all multi-GPU communication

### Tier 2: Stream-Ordered Memory Allocator (CUDA Native)

**CUDA Version Requirement:** CUDA 11.2+ for `cudaMallocAsync` / `cudaMemPool_t`

No new dependencies. The existing `MemoryPool` class (`include/cuda/memory/memory_pool.h`) wraps CUDA device memory, not stream-ordered allocator. A separate `DistributedMemoryPool` layer is needed for multi-GPU.

#### Key APIs

| API | Purpose | Docs |
|-----|---------|------|
| `cudaMallocAsync()` | Stream-ordered allocation | Programming Guide 15.3 |
| `cudaFreeAsync()` | Stream-ordered deallocation | Programming Guide 15.3 |
| `cudaMemPoolSetAccess()` | Set device accessibility for pool | Programming Guide 15.10 |
| `cudaMemPoolGetAccess()` | Query device access to a pool | Programming Guide 15.10 |
| `cudaMemPoolExportToShareableHandle()` | Export pool for IPC | Programming Guide 15.11.1 |
| `cudaMemPoolImportFromShareableHandle()` | Import shared pool | Programming Guide 15.11.2 |
| `cudaMemPoolSetAttribute()` | Configure reuse policies | Programming Guide 15.9 |

#### New Layer: Multi-GPU Memory Pool

```cpp
// New: include/cuda/memory/distributed_memory_pool.h
// Fits existing memory layer namespace: cuda::memory

class DistributedMemoryPool {
public:
    struct Config {
        size_t block_size = 1 << 20;
        size_t max_blocks_per_device = 16;
        bool enable_peer_access = true;
        bool share_across_devices = true;  // via cudaMemPoolSetAccess
    };

    // Initialize pools on all detected GPUs
    void initialize(const Config& config);
    
    // Allocate on specific device
    void* allocate(size_t bytes, int device_id, int stream_id = -1);
    
    // Allocate on device with most available memory
    void* allocate_auto(size_t bytes, int stream_id = -1);
    
    // Per-device pool metrics
    struct DevicePoolMetrics {
        int device_id;
        size_t allocated_bytes;
        size_t available_bytes;
        PoolMetrics local_pool;
    };
    std::vector<DevicePoolMetrics> get_all_metrics() const;
};
```

**Integration with existing layer:** The existing `MemoryPool` remains single-device. The new `DistributedMemoryPool` wraps per-device pools and exposes cross-device allocation. The existing `Buffer` and `unique_ptr` abstractions can be extended to support multi-device buffers.

**Integration with stream layer:** Stream-ordered allocation (`cudaMallocAsync`) uses the same `cudaStream_t` type as existing stream abstractions. Pass the stream from `cuda::async::StreamManager` directly.

**Key insight:** Set `cudaMemPoolAttr` `crossDevice` to allow allocations from one device's pool to be accessible from another. This handles the "distributed memory pool" requirement without any new libraries.

### Tier 3: NCCL — Only If Collectives Are Required

**NCCL Version:** NCCL 2.25+ (current stable), bundled with CUDA 17 as `libnccl.so`

**Dependency Decision: OPTIONAL, not required for v1.1 core.**

#### When NCCL Is Required

NCCL provides optimized implementations of:

| Operation | Use Case |
|-----------|----------|
| `ncclAllReduce` | Distributed gradient reduction, distributed batch norm |
| `ncclBroadcast` | Parameter broadcast, data parallel broadcast |
| `ncclAllGather` | Multi-GPU model output gathering |
| `ncclReduceScatter` | Ring-reduce patterns |
| `ncclSend`/`ncclRecv` | Point-to-point tensor exchange |
| `ncclGroupStart/End` | Batched collective launches |

NCCL uses CUDA Graph node APIs internally (`ncclRedOpCreatePreMulSum`, etc.) and integrates with CUDA streams natively.

#### When to Use NCCL vs. DIY

| Approach | Best For | Not Best For |
|----------|----------|--------------|
| **DIY with P2P** | 2-GPU data parallelism, simple reduce | 4+ GPUs, complex topologies |
| **NCCL** | 2-8 GPUs, all-reduce, broadcast, all-gather | Single GPU, trivial local-only |
| **NVSHMEM** | GPU-initiated RDMA, InfiniBand, multi-node | Single-node, no InfiniBand |

For MGPU-02 (multi-GPU data parallelism primitives) and MGPU-04 (multi-GPU matmul), NCCL is the industry-standard choice. However:

1. **For v1.1 initial release:** Ship with DIY P2P-based collectives (simple ring all-reduce via `cudaMemcpyAsyncPeer`) as a proof of concept. This avoids adding NCCL as a hard dependency.
2. **For v1.2:** Add NCCL as an optional dependency with a fallback to DIY implementations.

#### NCCL Integration

```cpp
// New: include/cuda/collective/multi_gpu_collectives.h
// New layer under algo: cuda::collective

class NCCLComm {
public:
    static void initialize(int ndevices);
    static void all_reduce(const void* send_buf, void* recv_buf, size_t count,
                           ncclDataType_t dtype, ncclRedOp_t op, cudaStream_t stream);
    static void broadcast(const void* send_buf, void* recv_buf, size_t count,
                          int root, cudaStream_t stream);
    // ... etc
};
```

**CMake integration:**

```cmake
# In CMakeLists.txt, add as optional:
find_library(NCCL_LIBRARY NAMES nccl)
if(NCCL_LIBRARY)
    set(NCCL_FOUND TRUE)
    set(NCCL_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nccl)
else()
    message(STATUS "NCCL not found — multi-GPU collectives will use fallback P2P implementation")
    set(NCCL_FOUND FALSE)
endif()
```

**NCCL and existing stream layer:** NCCL collective calls take a `cudaStream_t` parameter. Pass streams directly from `cuda::async::StreamManager`. NCCL's internal CUDA Graph integration is transparent to the caller.

### What NOT to Add

| Technology | Why Avoid for v1.1 |
|------------|-------------------|
| **NVSHMEM** | Targets GPU-initiated RDMA over InfiniBand/NVLink fabric. For single-node with CUDA IPC transport, NVSHMEM adds unnecessary daemon dependency, InfiniBand hardware requirement, and complexity. Revisit only for multi-node. |
| **CUDA MPS** | A system-level daemon for multi-process GPU sharing (e.g., MPI jobs). Not a library-level feature — it's a deployment configuration. The PROJECT.md explicitly says "single-node, not distributed multi-node." Not needed. |
| **GDRCopy** | For direct GPU-to-network RDMA. Requires specific hardware and kernel modules. Out of scope. |
| **UCX** | Communication framework for heterogeneous systems. Overkill for single-node multi-GPU with CUDA IPC. |

## Recommended Stack Additions

### New CMake Dependencies

```cmake
# CMakeLists.txt additions:

# 1. Already present — no change needed:
find_package(CUDAToolkit REQUIRED)

# 2. Optional NCCL (deferred to v1.2):
find_library(NCCL_LIBRARY NAMES nccl)
if(NCCL_LIBRARY)
    target_link_libraries(cuda_impl PUBLIC ${NCCL_LIBRARY})
    target_compile_definitions(cuda_impl PUBLIC NOVA_NCCL_ENABLED=1)
endif()
```

### New Directory Structure

```
include/cuda/
  +-- device/
  |     device_mesh.h          # NEW: DeviceMesh, peer detection
  |     device_utils.h         # existing
  +-- memory/
  |     distributed_memory_pool.h  # NEW: Multi-GPU pool
  |     memory_pool.h          # existing (single-device)
  +-- collective/              # NEW: Optional collective ops
        multi_gpu_collectives.h    # NCCL wrapper + P2P fallback
        reduce.h                     # multi-GPU reduce
        broadcast.h                  # multi-GPU broadcast

src/cuda/
  +-- device/
  |     device_mesh.cu         # NEW
  +-- collective/
  |     collective_ops.cu      # NEW: P2P fallback implementations
  |     nccl_ops.cu            # NEW: NCCL implementations (if enabled)
```

### New Targets in CMakeLists.txt

```cmake
set(MULTI_GPU_SOURCES
    ${CMAKE_SOURCE_DIR}/src/cuda/device/device_mesh.cu
    ${CMAKE_SOURCE_DIR}/src/cuda/collective/collective_ops.cu
)

if(NCCL_FOUND)
    list(APPEND MULTI_GPU_SOURCES
        ${CMAKE_SOURCE_DIR}/src/cuda/collective/nccl_ops.cu
    )
endif()

# Add to cuda_impl sources:
set(ALL_CUDA_SOURCES
    ${DEVICE_SOURCES}
    ${ALGO_SOURCES}
    # ... existing
    ${MULTI_GPU_SOURCES}       # ADD THIS
)
```

## Integration Points with Existing Architecture

### Layer: Device (MGPU-01: Device Mesh)

The `DeviceMesh` class extends the existing `device` layer:

```
Existing: cuda::device namespace
  - ReduceOp, warp_reduce, block_reduce
  - Error handling (CUDA_CHECK, CudaException)

New additions:
  - DeviceMesh class: detect_mesh(), enable_peer_access(), can_access()
  - Peer memory copy helpers
```

### Layer: Memory (MGPU-03: Distributed Memory Pool)

The `DistributedMemoryPool` extends the existing `memory` layer:

```
Existing: cuda::memory namespace
  - Buffer, unique_ptr, MemoryPool
  - PoolMetrics, defragment()

New additions:
  - DistributedMemoryPool: per-device pools with cross-device visibility
  - DevicePoolMetrics for per-GPU reporting
  - Auto-allocation based on device memory availability
```

### Layer: Algo (MGPU-02, MGPU-04)

New `cuda::collective` sub-namespace under `algo`:

```
Existing: cuda::algo namespace
  - KernelLauncher, reduce()

New additions:
  - cuda::collective::multi_gpu_reduce (all-reduce, ring-reduce)
  - cuda::collective::multi_gpu_broadcast
  - cuda::collective::multi_gpu_all_gather
  - cuda::collective::distributed_matmul (uses above primitives)
```

### Integration with Async/Stream Layer

The existing `cuda::async::StreamManager` works unchanged. Multi-GPU operations accept `cudaStream_t`:

```cpp
// Existing API works as-is:
auto stream = cuda::async::global_stream_manager().get_stream(0);

// New multi-GPU ops accept the same stream:
cuda::collective::all_reduce(send, recv, count, dtype, op, stream);
distributed_pool.allocate(bytes, device_id, stream);  // stream optional
```

No changes to the stream abstraction layer are needed.

## Memory Pool Changes Summary

| Change | Location | What to Modify |
|--------|----------|----------------|
| Stream-ordered allocation | `memory/memory_pool.h` | Add `StreamOrderedPool` wrapper class |
| Multi-device pool | New `memory/distributed_memory_pool.h` | New class, wraps per-device pools |
| Peer-access config | `memory/buffer.h` | Add `device_id` field, peer-aware copy ctor |
| Metrics extension | `memory/memory_pool.h` `PoolMetrics` | Add `device_id`, `peer_access_bytes` |
| `unique_ptr` extension | `memory/unique_ptr.h` | Add `DeviceBuffer` variant with device_id |

## Testing Implications

| New Component | Test Coverage Needed |
|---------------|---------------------|
| `DeviceMesh::detect_mesh()` | Test on 1, 2, 4, 8 GPU configurations |
| `DeviceMesh::enable_peer_access()` | Test enable/disable cycles, error on unsupported P2P |
| `DistributedMemoryPool` | Allocate from each device, cross-device access |
| Collective ops | Numerical correctness vs. single-GPU reference |
| Multi-GPU matmul | Accuracy vs. single-GPU matmul (existing tests) |

Existing 81+ tests should not regress. Multi-GPU tests require multi-GPU hardware — use Google Test's `TYPED_TEST` with a `__CUDA_ARCH__` or device-count skip.

## Source Confidence

| Technology | Confidence | Notes |
|------------|------------|-------|
| CUDA Peer APIs | HIGH | CUDA Programming Guide 6.2.9, 15.10-15.11 |
| Stream-ordered allocator | HIGH | CUDA Programming Guide 15, available since CUDA 11.2 |
| IPC Memory Pools | HIGH | CUDA Programming Guide 15.11 |
| NCCL | MEDIUM | NVIDIA official docs; CNMEM-bundled with CUDA 17 |
| DIY P2P collectives | MEDIUM | Standard ring-allreduce pattern, well-documented |
| NVSHMEM exclusion | HIGH | Docs confirm hardware/transport requirements |
| CUDA MPS exclusion | HIGH | Project constraint, single-node only |
