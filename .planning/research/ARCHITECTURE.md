# Architecture Research: Multi-GPU Support for Nova CUDA Library

**Domain:** CUDA C++ multi-GPU parallel compute library
**Researched:** 2026-04-24
**Confidence:** HIGH

## Executive Summary

Multi-GPU support integrates into Nova's five-layer architecture by (1) extending the existing memory and device layers with device-mesh awareness and peer memory primitives, (2) inserting a new peer transport layer between device and algo, and (3) adding distributed ops as new components in the algo layer. The design is additive -- all existing single-GPU code remains unchanged. Key architectural decisions: use explicit multi-GPU API over auto-parallelism, make NCCL an optional dependency, and integrate mesh topology discovery into the existing `cuda::performance` namespace.

---

## Standard Architecture

### System Overview -- Extended Six-Layer Model

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: High-Level Distributed API (STL-style)            │
│  cuda::distributed::reduce(), cuda::distributed::matmul()  │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Distributed Operations (NEW)                      │
│  DistributedReduce, DistributedBroadcast, MultiGPUMatmul   │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Algorithm Wrappers                                │
│  cuda::algo::reduce_sum(), memory management               │
│  ─────────────────────────────────────────────────────────  │
│  Layer 2.5: Peer Transport (NEW -- between device and algo)│
│  cuda::mesh::PeerCopy, cuda::mesh::Barrier                 │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Device Kernels                                   │
│  Pure __global__ kernels, no memory allocation             │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Memory Foundation (EXTENDED)                      │
│  Buffer<T>, unique_ptr<T>, MemoryPool, PeerMemoryPool      │
│  cuda::memory::PeerAllocator                               │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Device Abstraction (EXTENDED)                     │
│  DeviceInfo, DeviceMesh, PeerCapabilityMap                 │
│  cuda::device, cuda::mesh                                   │
└─────────────────────────────────────────────────────────────┘
```

### Layer Renumbering Rationale

The existing five layers (memory, device, algo, api, plus cross-cutting async) map to the new model as follows:

| Existing Layer | New Mapping | Change |
|----------------|-------------|--------|
| Layer 0: Memory Foundation | Layer 0 + Peer Extensions | EXTENDED |
| Layer 1: Device Kernels | Layer 2: Device Kernels | UNCHANGED |
| Layer 2: Algorithm Wrappers | Layer 3: Algorithm Wrappers | UNCHANGED |
| Layer 3: High-Level API | Layer 5: Distributed API | EXTENDED |
| async/streaming cross-cutting | Layer 2.5: Peer Transport | NEW LAYER |

**Decision:** Number the new peer transport layer as 2.5 rather than inserting it as "Layer 3" to minimize changes to existing header includes and macro guards.

### Component Responsibilities

| Component | Layer | Responsibility | File Location |
|-----------|-------|----------------|---------------|
| `DeviceMesh` | 0 (ext) | Device enumeration, topology, peer capability matrix | `include/cuda/mesh/device_mesh.h` |
| `PeerCapabilityMap` | 0 (ext) | Caches `cudaDeviceCanAccessPeer` results | `include/cuda/mesh/device_mesh.h` |
| `PeerAllocator` | 1 (ext) | Memory allocation with peer accessibility hints | `include/cuda/memory/peer_allocator.h` |
| `DistributedMemoryPool` | 1 (ext) | Multi-GPU memory pool with cross-device allocation | `include/cuda/memory/distributed_pool.h` |
| `PeerCopy` | 2.5 | Async peer-to-peer memory copy via StreamManager | `include/cuda/mesh/peer_copy.h` |
| `MeshBarrier` | 2.5 | Multi-GPU synchronization via events | `include/cuda/mesh/mesh_barrier.h` |
| `DistributedReduce` | 4 (new) | Multi-GPU reduction using tree or ring algorithm | `include/cuda/distributed/reduce.h` |
| `DistributedBroadcast` | 4 (new) | Multi-GPU broadcast to all ranks | `include/cuda/distributed/broadcast.h` |
| `DistributedMatmul` | 4 (new) | Multi-GPU matmul (data-parallel or tensor-parallel) | `include/cuda/distributed/matmul.h` |
| `StreamManager` | cross | Existing -- extended with per-device streams | `include/cuda/async/stream_manager.h` |

---

## Recommended Project Structure

```
include/cuda/
    mesh/                          # NEW: Device mesh layer
    │   ├── device_mesh.h         # DeviceMesh, PeerCapabilityMap
    │   ├── peer_copy.h           # PeerCopy async copy primitive
    │   └── mesh_barrier.h        # MeshBarrier sync primitive
    memory/
    │   ├── peer_allocator.h      # NEW: Peer-aware allocator
    │   └── distributed_pool.h    # NEW: Distributed memory pool
    distributed/                   # NEW: Distributed ops layer
    │   ├── reduce.h              # DistributedReduce
    │   ├── broadcast.h           # DistributedBroadcast
    │   ├── barrier.h             # DistributedBarrier
    │   └── matmul.h              # DistributedMatmul

src/cuda/
    mesh/                          # NEW: Mesh implementation
    │   ├── device_mesh.cu
    │   ├── peer_copy.cu
    │   └── mesh_barrier.cu
    distributed/                   # NEW: Distributed impl
    │   ├── reduce.cu
    │   ├── broadcast.cu
    │   └── matmul.cu
    memory/
    │   ├── peer_allocator.cpp    # NEW
    │   └── distributed_pool.cpp  # NEW
```

### Structure Rationale

- **`mesh/`**: Groups all device topology and peer access code. Mirrors the existing `performance/` directory which also extends the device layer.
- **`memory/peer_allocator.h`**: Extends the existing memory layer. Placed alongside `memory_pool.h` to make the parallel obvious.
- **`distributed/`**: New top-level module for collective-style operations. Not placed inside `algo/` because distributed ops span multiple devices and have fundamentally different semantics than single-device algos.
- **`.cpp` files for memory components**: Memory allocation and pool management belong in `.cpp` (not `.cu`) because they contain no device code and enable better compile-time separation.

---

## Architectural Patterns

### Pattern 1: Device-Mesh Singleton

**What:** A global `DeviceMesh` instance that lazily initializes peer access between all available GPUs on first access.

**When to use:** When you need a consistent view of the device topology across the entire application.

**Trade-offs:**
- Pros: Single source of truth, lazy initialization avoids probing GPUs that are never used.
- Cons: Global mutable state; testing requires mocking or resetting the singleton.

**Example:**
```cpp
// include/cuda/mesh/device_mesh.h
namespace cuda::mesh {

class DeviceMesh {
public:
    static DeviceMesh& instance();

    int device_count() const { return device_count_; }
    bool can_access_peer(int src, int dst) const;
    void enable_peer_access(int src, int dst);

    // Returns a list of all devices that can communicate directly
    std::vector<int> get_mesh_devices() const;

    // Partition data across devices
    std::pair<size_t, size_t> get_partition(size_t global_size, int rank) const;

    DeviceMesh(const DeviceMesh&) = delete;
    DeviceMesh& operator=(const DeviceMesh&) = delete;

private:
    DeviceMesh();
    int device_count_ = 0;
    std::vector<std::vector<bool>> peer_matrix_;  // peer_matrix_[i][j] = can i access j
    std::vector<int> mesh_devices_;
};

}  // namespace cuda::mesh
```

### Pattern 2: Per-Device RAII Scoped Context

**What:** Each per-device operation uses a `ScopedDevice` RAII guard that saves/restores the current device.

**When to use:** Any multi-GPU operation that iterates over devices.

**Trade-offs:**
- Pros: Exception-safe, composable with existing device management.
- Cons: Adds overhead for simple single-GPU paths.

**Example:**
```cpp
// include/cuda/mesh/scoped_device.h
namespace cuda::mesh {

class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&saved_device_));
        CUDA_CHECK(cudaSetDevice(device));
    }
    ~ScopedDevice() { cudaSetDevice(saved_device_); }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int saved_device_;
};

}  // namespace cuda::mesh
```

### Pattern 3: Collective Operation with Event-Based Synchronization

**What:** Distributed ops use CUDA events on per-device streams to synchronize across GPUs without host blocking.

**When to use:** Multi-GPU collectives where you want to overlap communication and computation.

**Trade-offs:**
- Pros: Non-blocking, pipelined execution.
- Cons: More complex than host-synchronized alternatives; event ordering must be carefully managed.

**Example:**
```cpp
// Pseudo-code for DistributedReduce::execute()
template <typename T, typename ReduceOp>
T DistributedReduce::execute(const T* local_data, size_t count, ReduceOp op) {
    // Phase 1: Local reduction on each device
    launch_local_reduce_kernel(local_data, count, op);

    // Phase 2: Ring reduction using peer copies
    for (int step = 1; step < mesh_size_; step <<= 1) {
        int peer = (rank_ - step + mesh_size_) % mesh_size_;

        // Emit async peer copy (from peer -> receive buffer)
        CUDA_CHECK(cudaMemcpyAsync(
            recv_buffers_[step % 2].data(),
            peer_buffers_[peer].data(),
            count * sizeof(T),
            cudaMemcpyDeviceToDevice,
            streams_[peer].get()
        ));

        // Emit event to signal copy completion
        CUDA_CHECK(cudaEventRecord(events_[peer].get(), streams_[peer].get()));

        // Wait for peer's event on our stream
        CUDA_CHECK(cudaStreamWaitEvent(streams_[rank_].get(), events_[peer].get(), 0));

        // Reduce received data with local result
        launch_elementwise_op(receive_buffer, local_buffer, count, op);
    }
    return final_result_;
}
```

### Pattern 4: Stream-per-Device in StreamManager Extension

**What:** `StreamManager` is extended to manage one stream per device, enabling concurrent multi-device operations.

**When to use:** Any multi-GPU algorithm that needs independent streams for each device.

**Example (extension to existing `stream_manager.h`):**
```cpp
namespace cuda::async {

class MultiDeviceStreamManager {
public:
    MultiDeviceStreamManager() {
        int count = cuda::performance::get_device_count();
        per_device_streams_.resize(count);
        for (int d = 0; d < count; ++d) {
            CUDA_CHECK(cudaSetDevice(d));
            per_device_streams_[d] = std::make_unique<cuda::stream::Stream>(0, cudaStreamNonBlocking);
        }
    }

    cudaStream_t get_stream_for_device(int device) {
        return per_device_streams_.at(device)->get();
    }

    void synchronize_all_devices() {
        for (auto& s : per_device_streams_) {
            s->synchronize();
        }
    }

private:
    std::vector<std::unique_ptr<cuda::stream::Stream>> per_device_streams_;
};

}  // namespace cuda::async
```

---

## Data Flow

### Multi-GPU Data Parallel Reduction

```
Host: cuda::distributed::reduce(data, count, op)
     │
     ▼
DeviceMesh::instance()          [Layer 0 - Device Mesh]
     │
     ▼
DistributedReduce::partition()  [Layer 4 - Distributed Ops]
     │   Partition global data into chunks per device
     │
     ├──────────────────────────────┐
     ▼                              ▼
Device 0 Chunk                Device 1 Chunk
     │                              │
     ▼                              ▼
LocalReduce kernel            LocalReduce kernel
     │                              │
     ├──────────────────────────────┤
     │        MeshBarrier           │
     │   (sync all devices)         │
     ├──────────────────────────────┤
     ▼                              ▼
Ring PeerCopy (step 1)  ───►  Ring PeerCopy (step 1)
     │                              │
     ▼                              ▼
Cross-reduce kernel           Cross-reduce kernel
     │                              │
     ▼                              ▼
Ring PeerCopy (step 2)  ───►  Ring PeerCopy (step 2)
     │
     ▼
Final result on device 0
```

### Multi-GPU Matmul (Data Parallel)

```
Input: [A|B] split across N devices along rows
     │
     ▼
Per-device: A_i @ B_i -> C_i       [Layer 4 - DistributedMatmul]
     │
     ├──────────────────────────────┐
     ▼                              ▼
Device 0: C_0                  Device 1: C_1
     │                              │
     ▼                              ▼
AllGather partial results      AllGather partial results
     │
     ▼
Output: C = concat(C_0, C_1, ...)
```

---

## Integration Points

### Integration with Existing Components

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `cuda::performance::DeviceInfo` -> `DeviceMesh` | Composition | `DeviceMesh` uses `get_device_count()`, `get_device_properties()` internally |
| `cuda::async::StreamManager` -> `PeerCopy` | Stream handles | `PeerCopy` accepts `cudaStream_t` from `StreamManager` |
| `cuda::memory::MemoryPool` -> `DistributedMemoryPool` | Inheritance | `DistributedMemoryPool` wraps N `MemoryPool` instances, one per device |
| `cuda::neural::matmul` -> `DistributedMatmul` | Wrapper pattern | `DistributedMatmul` calls `cuda::neural::matmul` per device |
| `cuda::device::CublasContext` -> `DistributedMatmul` | Per-device handles | Each device needs its own `cublasHandle_t` |

### Backward Compatibility

**All existing single-GPU code is unaffected** because:
1. `DeviceMesh::instance()` is lazily constructed on first multi-GPU API call.
2. Existing `cuda::memory::Buffer<T>`, `cuda::memory::MemoryPool`, `cuda::neural::matmul` remain in-place.
3. No existing headers are modified -- all multi-GPU types are in new headers.
4. CMake library targets are additive (new targets `cuda_mesh`, `cuda_distributed`).

**API design principle:** Explicit opt-in. Users must call `cuda::distributed::reduce()` or construct a `DeviceMesh` to trigger multi-GPU behavior.

---

## Build System Changes

### CMake Modifications

```cmake
# New directories
set(CUDA_MESH_DIR ${CMAKE_SOURCE_DIR}/include/cuda/mesh)
set(CUDA_DISTRIBUTED_DIR ${CMAKE_SOURCE_DIR}/include/cuda/distributed)
set(SRC_CUDA_MESH ${CMAKE_SOURCE_DIR}/src/cuda/mesh)
set(SRC_CUDA_DISTRIBUTED ${CMAKE_SOURCE_DIR}/src/cuda/distributed)

# Optional NCCL dependency
option(NOVA_ENABLE_NCCL "Enable NCCL for multi-GPU collectives" OFF)
if(NOVA_ENABLE_NCCL)
    find_package(NCCL 2.18 QUIET)
    if(NOT NCCL_FOUND)
        message(WARNING "NCCL not found. Multi-GPU collectives will use CUDA-native fallback.")
    endif()
endif()

# New interface libraries
add_library(cuda_mesh INTERFACE)
target_include_directories(cuda_mesh INTERFACE
        $<BUILD_INTERFACE:${CUDA_MESH_DIR}>
)
target_link_libraries(cuda_mesh INTERFACE cuda_device CUDA::cudart)

add_library(cuda_distributed INTERFACE)
target_include_directories(cuda_distributed INTERFACE
        $<BUILD_INTERFACE:${CUDA_DISTRIBUTED_DIR}>
)
target_link_libraries(cuda_distributed INTERFACE cuda_mesh cuda_algo)
if(NCCL_FOUND)
    target_link_libraries(cuda_distributed INTERFACE NCCL::nccl)
    target_compile_definitions(cuda_distributed INTERFACE NOVA_NCCL_ENABLED=1)
else()
    target_compile_definitions(cuda_distributed INTERFACE NOVA_NCCL_ENABLED=0)
endif()

# New source groups
set(MESH_SOURCES
        ${SRC_CUDA_MESH}/device_mesh.cu
        ${SRC_CUDA_MESH}/peer_copy.cu
        ${SRC_CUDA_MESH}/mesh_barrier.cu
)

set(DISTRIBUTED_SOURCES
        ${SRC_CUDA_DISTRIBUTED}/reduce.cu
        ${SRC_CUDA_DISTRIBUTED}/broadcast.cu
        ${SRC_CUDA_DISTRIBUTED}/matmul.cu
)

# New static library
add_library(cuda_multigpu STATIC ${MESH_SOURCES} ${DISTRIBUTED_SOURCES})
target_include_directories(cuda_multigpu PUBLIC
        ${CUDA_MESH_DIR}
        ${CUDA_DISTRIBUTED_DIR}
)
target_link_libraries(cuda_multigpu PUBLIC
        cuda_impl
        cuda_mesh
        cuda_distributed
)
if(NCCL_FOUND)
    target_link_libraries(cuda_multigpu PUBLIC NCCL::nccl)
endif()
set_target_properties(cuda_multigpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
```

### Key Build Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| NCCL as optional (`find_package` + fallback) | Not all systems have NCCL; CUDA-native collectives work for simple ops | Simpler ops available without NCCL; complex ops may degrade |
| New `cuda_multigpu` static lib | Separates compilation; faster incremental builds for single-GPU users | Extra link step for multi-GPU users |
| `CUDA_SEPARABLE_COMPILATION` on new lib | Matches existing `cuda_impl` pattern | Required for proper device code separation |

---

## Anti-Patterns

### Anti-Pattern 1: Implicit Device Switching

**What people do:** Call `cudaSetDevice()` inside library functions without saving/restoring the previous device.

**Why it's wrong:** Breaks the caller's CUDA state, causing silent correctness bugs when the caller expects to remain on their device.

**Do this instead:** Use `ScopedDevice` RAII guard. Always restore the previous device.

### Anti-Pattern 2: Global Peer Access Enabling

**What people do:** Call `cudaDeviceEnablePeerAccess()` for all device pairs at library init time.

**Why it's wrong:** Peer access consumes resources (PCIe bandwidth, memory mapped I/O). Enabling for devices that will never communicate wastes resources. Also, not all device pairs support peer access -- the matrix is sparse on multi-host systems.

**Do this instead:** Query `cudaDeviceCanAccessPeer()` first, then enable peer access lazily per actual communication pair. Cache the results in `PeerCapabilityMap`.

### Anti-Pattern 3: Single Stream for Multi-Device Ops

**What people do:** Use a single CUDA stream across multiple devices.

**Why it's wrong:** Streams are device-local. A `cudaStream_t` is only valid on the device that created it.

**Do this instead:** Maintain one `cudaStream_t` per device. Use `MultiDeviceStreamManager`.

### Anti-Pattern 4: Blocking Host Synchronization in Collectives

**What people do:** Call `cudaDeviceSynchronize()` or `cudaStreamSynchronize()` inside multi-GPU collective implementations.

**Why it's wrong:** Blocks all work on that device, preventing overlap with peer operations. Causes deadlocks in cyclic communication patterns.

**Do this instead:** Use event-based synchronization (`cudaEventRecord` + `cudaStreamWaitEvent`) which is non-blocking and compositional.

---

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1 GPU | Existing single-GPU path. No multi-GPU overhead. |
| 2-4 GPUs (single-node) | Full peer access available. Ring-based collectives. Simple topology. |
| 4-8 GPUs (single-node) | PCIe/NVLink topology matters. Need `PeerCapabilityMap` to pick optimal communication paths. Consider NCCL for complex collectives. |
| Multi-node | Out of scope for v1.1. Design the `DistributedMatmul` interface to be swappable for multi-node backends. |

### Scaling Priorities

1. **First bottleneck (2-4 GPUs):** Peer memory bandwidth -- use `cudaMemcpyAsync` with non-blocking streams to overlap copies with compute.
2. **Second bottleneck (4+ GPUs):** Collective algorithm selection -- ring vs tree reduction. Start with ring (simpler), add tree optimization when profiling shows it matters.
3. **Third bottleneck (heterogeneous GPUs):** Different compute capabilities in mesh. Handle via device-specific block size selection and potential kernel specialization.

---

## Phase-Specific Architecture Guidance

### Phase: Device Mesh Detection (MGPU-01)

**Focus:** `DeviceMesh`, `PeerCapabilityMap`, `ScopedDevice`

**Architecture approach:**
- Create `include/cuda/mesh/device_mesh.h` with the singleton `DeviceMesh`.
- Query all device pairs with `cudaDeviceCanAccessPeer()` and cache results.
- Add `ScopedDevice` for exception-safe device switching.
- Extend `cuda::performance::DeviceInfo` with peer-related methods (optional -- or keep in `DeviceMesh` to preserve header isolation).

**Integration points:**
- Uses: `cuda::performance::get_device_count()`, `get_device_properties()`
- Extended by: `PeerCopy`, `DistributedMemoryPool`

**CMake:** Add `cuda_mesh` interface library. No new `.cu` files required for pure host-side topology discovery.

### Phase: Peer Memory & Async Peer Copies (MGPU-02)

**Focus:** `PeerAllocator`, `DistributedMemoryPool`, `PeerCopy`

**Architecture approach:**
- `PeerAllocator`: Subclass or companion to `MemoryPool` that allocates memory accessible via peer paths.
- `DistributedMemoryPool`: Per-device pool instances managed by a coordinator. Allocation strategy: allocate on device closest to the data or round-robin.
- `PeerCopy`: Thin wrapper around `cudaMemcpyAsync` with peer memory, integrated with `StreamManager`.

**Integration points:**
- Uses: `DeviceMesh::can_access_peer()`, `StreamManager::get_stream()`
- Extended by: `DistributedReduce`, `DistributedMatmul`

**CMake:** Add source files to `cuda_multigpu` library. Depends on `cuda_mesh`.

### Phase: Distributed Ops Primitives (MGPU-02 continued)

**Focus:** `DistributedReduce`, `DistributedBroadcast`, `DistributedBarrier`

**Architecture approach:**
- All three use the same underlying pattern: event-based synchronization with per-device streams.
- Ring reduction is the default algorithm (simpler, no coordinator bottleneck).
- `DistributedBarrier` uses a central counter in GPU global memory with atomic operations, or relies on NCCL barrier when available.

**Integration points:**
- Uses: `MeshBarrier`, `PeerCopy`, `DistributedMemoryPool`
- Extended by: `DistributedMatmul`

**CMake:** Add to `cuda_multigpu`. Optional NCCL backend via `#ifdef NOVA_NCCL_ENABLED`.

### Phase: Multi-GPU Matmul (MGPU-04)

**Focus:** `DistributedMatmul`

**Architecture approach:**
- Start with **data parallelism** (row-partitioned A matrix). This is the simplest and maps directly to existing `cuda::neural::matmul`.
- Each device computes `C_i = A_i @ B`. Results are then reduced or gathered depending on the output requirement.
- **Tensor parallelism** and **pipeline parallelism** are deferred until after the data-parallel baseline works. The `DistributedMatmul` interface should abstract the parallelism strategy.

**Interface design:**
```cpp
namespace cuda::distributed {

enum class ParallelismStrategy {
    DataParallel,   // Row-partition A across GPUs
    TensorParallel, // Column-partition weights (future)
    PipelineParallel // Micro-batch pipeline (future)
};

struct DistributedMatmulOptions {
    ParallelismStrategy strategy = ParallelismStrategy::DataParallel;
    bool all_reduce_output = true;  // vs. partitioned output
    bool use_nccl = false;          // CUDA-native if false
};

void matmul(
    const float* A,   // Full matrix (or partitioned if use_partitioned_input=true)
    const float* B,
    float* C,
    int m, int n, int k,
    DistributedMatmulOptions opts = {}
);

}  // namespace cuda::distributed
```

**Integration points:**
- Uses: `cuda::neural::matmul` per device, `DistributedReduce`, `DistributedBroadcast`
- Requires: Per-device `cublasHandle_t` (extend `CublasContext` or use `get_cublas_handle()` per device)

---

## Sources

- **CUDA C++ Programming Guide (Section 6.2.9 - Multi-Device System)**: Device enumeration, peer-to-peer memory access, peer-to-peer memory copy. NVIDIA, 2026. [URL](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system)
- **CUDA C++ Programming Guide (Section 15 - Stream Ordered Memory Allocator, 15.10 - Device Accessibility for Multi-GPU Support)**: Memory pools across devices, cudaMemPool_t per-device access. NVIDIA, 2026.
- **CUDA C++ Programming Guide (Section 16.7 - Peer Access)**: Graph node APIs for peer access and stream capture. NVIDIA, 2026.
- **Existing Nova codebase**: Layer structure, `MemoryPool`, `StreamManager`, `DeviceInfo`, `Buffer<T>`, `CublasContext` patterns.
- **Cooperative Groups (Section 11)**: Multi-GPU synchronization patterns and collectives. NVIDIA, 2026.

---

*Architecture research for: Nova CUDA Library v1.1 Multi-GPU Support*
*Researched: 2026-04-24*
