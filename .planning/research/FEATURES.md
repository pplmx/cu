# Feature Landscape: Multi-GPU Support

**Domain:** CUDA GPU compute library — single-node multi-GPU data parallelism
**Researched:** 2026-04-24
**Confidence:** HIGH (based on NVIDIA CUDA Programming Guide, NCCL 2.30 documentation)

---

## Executive Summary

Multi-GPU support in a CUDA library requires a layered approach: foundations first (device mesh, peer access), then primitives (collectives, synchronization), then distributed operations (multi-GPU matmul). The critical insight is that peer-to-peer memory access is prerequisite to all efficient multi-GPU operations. Libraries like NCCL provide battle-tested implementations for collectives but require careful integration with the library's existing stream and memory management infrastructure.

---

## Feature Categories

### Category 1: Device Mesh Management

**Goal:** Discover and represent GPU topology for intelligent data placement and communication.

#### 1.1 GPU Detection and Enumeration

| Aspect | Details |
|--------|---------|
| **What it does** | Enumerate available CUDA devices, query properties (memory, compute capability, clock rates) |
| **CUDA API** | `cudaGetDeviceCount()`, `cudaGetDeviceProperties()` |
| **Why expected** | Any multi-GPU library must know what hardware exists |
| **Complexity** | Low |
| **Dependencies** | None |

**Table Stakes?** YES — Foundation for all other multi-GPU features.

**API Surface:**
```cpp
namespace cuda::mesh {

struct DeviceInfo {
    int device_id;
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multi_processor_count;
    // ... additional properties
};

std::vector<DeviceInfo> enumerate_devices();
int get_device_count();

}
```

#### 1.2 Peer Access Capability Queries

| Aspect | Details |
|--------|---------|
| **What it does** | Query whether direct GPU-to-GPU memory access is possible between device pairs |
| **CUDA API** | `cudaDeviceCanAccessPeer()`, `cudaDeviceEnablePeerAccess()` |
| **Why expected** | Determines if peer memory access is available; fallback to host-mediated copy if not |
| **Complexity** | Low |
| **Dependencies** | GPU enumeration |

**Table Stakes?** YES — Required before attempting direct peer access.

**API Surface:**
```cpp
namespace cuda::mesh {

struct PeerAccessInfo {
    int source_device;
    int target_device;
    bool can_access_peer;
    // P2P bandwidth can be queried via benchmarks
};

std::vector<PeerAccessInfo> query_peer_access(const std::vector<DeviceInfo>& devices);
void enable_peer_access(int src, int dst);  // throws if not supported

}
```

#### 1.3 Device Mesh Topology Representation

| Aspect | Details |
|--------|---------|
| **What it does** | Build a topology graph showing interconnect (NVLink, PCIe, CPU) between GPUs |
| **CUDA API** | `cudaDeviceGetP2PAttribute()` for bandwidth, `nvmlDeviceGetTopology()` via NVML |
| **Why expected** | Enables optimal collective algorithm selection (ring vs tree) and data placement |
| **Complexity** | Medium |
| **Dependencies** | Peer access queries |

**Table Stakes?** PARTIAL — Critical for performance optimization, optional for basic functionality.

**Differentiator?** YES — Most libraries expose raw peer access; topology-aware collectives are a quality signal.

**API Surface:**
```cpp
namespace cuda::mesh {

enum class InterconnectType {
    NVLink,
    PCIe,
    QSFP,     // InfiniBand/NVLink over fabric
    Unknown
};

struct DeviceMesh {
    std::vector<DeviceInfo> devices;
    // adjacency matrix or list of connections
    std::vector<std::vector<InterconnectType>> topology;
    
    // Query optimal placement for a given data size
    std::vector<int> suggest_placement(size_t total_bytes_per_device);
};

DeviceMesh build_topology();

}
```

#### 1.4 CUDA MPS Management

| Aspect | Details |
|--------|---------|
| **What it does** | Manage NVIDIA Multi-Process Service for GPU sharing between processes |
| **Background** | CUDA MPS allows multiple processes to share a single GPU with controlled resource allocation |
| **When to use** | Single-process multi-GPU is primary; MPS is fallback for multi-process scenarios |
| **Complexity** | Medium |
| **Dependencies** | None |

**Table Stakes?** NO — Single-process multi-GPU is sufficient for v1.1 scope.
**Differentiator?** NO — Defer to future work if multi-process scenarios emerge.

---

### Category 2: Multi-GPU Memory Operations

**Goal:** Efficient data movement across GPUs with peer access optimization.

#### 2.1 Peer Memory Access (Direct GPU-to-GPU)

| Aspect | Details |
|--------|---------|
| **What it does** | Allow kernel on GPU A to directly read/write memory allocated on GPU B |
| **CUDA API** | `cudaDeviceEnablePeerAccess()`, then kernels can access peer device memory directly |
| **Performance** | 2-3x faster than host-mediated PCIe transfers for large contiguous accesses |
| **Limitations** | Limited by NVLink/PCIe bandwidth; not all GPU pairs have direct access |
| **Complexity** | Medium |
| **Dependencies** | Peer access capability queries |

**Table Stakes?** YES — Foundation for efficient multi-GPU compute.

**API Surface:**
```cpp
namespace cuda::memory {

class PeerMemoryRegion {
public:
    PeerMemoryRegion(int src_device, int dst_device);
    ~PeerMemoryRegion();
    
    // Direct kernel access via peer pointers
    template<typename T>
    const T* get_peer_pointer(const T* local_ptr, int remote_device) const;
    
    bool is_accessible() const { return accessible_; }
    
private:
    int src_, dst_;
    bool accessible_;
};

// RAII wrapper for enabling peer access
class PeerAccessGuard {
public:
    PeerAccessGuard(int src, int dst);
    ~PeerAccessGuard();
    
private:
    int src_, dst_;
};

}
```

#### 2.2 Distributed Memory Pool Across GPUs

| Aspect | Details |
|--------|---------|
| **What it does** | Extend the single-GPU memory pool (existing in v1.0) to span multiple GPUs with coherent allocation |
| **Existing patterns** | Nova's v1.0 `MemoryPool` with stream-aware allocation and defragmentation |
| **Challenge** | Allocation must respect device locality; deallocation must handle cross-GPU references |
| **Complexity** | High |
| **Dependencies** | Device mesh topology, peer access |

**Table Stakes?** YES — Required for multi-GPU workloads with dynamic memory patterns.

**Design Considerations:**
```cpp
namespace cuda::memory {

class DistributedMemoryPool {
public:
    struct Config {
        size_t block_size = 1 << 20;
        size_t max_blocks_per_device = 16;
        bool enable_peer_access = true;  // use direct P2P when available
    };
    
    // Allocate from local device
    void* allocate(size_t bytes, int device_id, int stream_id = -1);
    
    // Allocate on remote device (for peer access patterns)
    void* allocate_remote(size_t bytes, int local_device, int remote_device);
    
    // Query which device holds a pointer
    int locate_device(const void* ptr) const;
    
    PoolMetrics get_metrics(int device_id) const;
    void defragment(int device_id);
    
private:
    std::vector<MemoryPool> pools_;  // one per device
    std::unordered_map<const void*, int> ptr_to_device_;
};

}
```

#### 2.3 Unified Memory Considerations (Managed Memory)

| Aspect | Details |
|--------|---------|
| **What it does** | Use CUDA managed memory (`cudaMallocManaged`) for automatic page migration |
| **Trade-offs** | Simpler programming model but page fault overhead; best for irregular access patterns |
| **When to use** | When data access patterns are unpredictable; disable for bulk transfers |
| **Complexity** | Medium |
| **Dependencies** | None (orthogonal feature) |

**Table Stakes?** NO — Explicit peer copies are more predictable.
**Differentiator?** NO — Defer unless user demand emerges.

#### 2.4 Async Peer-to-Peer Copies

| Aspect | Details |
|--------|---------|
| **What it does** | Asynchronous GPU-to-GPU memory copy using CUDA streams |
| **CUDA API** | `cudaMemcpyAsync()` with peer-to-peer addressing enabled |
| **Integration** | Must work with existing `StreamManager` and priority streams |
| **Complexity** | Medium |
| **Dependencies** | Peer access capability queries |

**Table Stakes?** YES — Core building block for data movement.

**API Surface:**
```cpp
namespace cuda::memory {

// Async copy between devices
void memcpy_peer_async(
    void* dst, int dst_device,
    const void* src, int src_device,
    size_t bytes,
    cudaStream_t stream
);

// Convenience wrapper respecting peer access
void memcpy_peer_async(
    void* dst, const void* src,
    size_t bytes,
    cudaStream_t stream
);  // infers devices from pointers

}
```

---

### Category 3: Data Parallelism Primitives

**Goal:** Provide NCCL-style collective operations for multi-GPU synchronization and data movement.

#### 3.1 Multi-GPU Reduce (All-Reduce, Reduce-Scatter)

| Aspect | Details |
|--------|---------|
| **What it does** | Reduce values across GPUs (sum, min, max, etc.) with result distributed appropriately |
| **Implementation options** | NCCL (production) vs custom (educational, learning) |
| **Key algorithms** | Ring-reduce, tree-reduce, recursive doubling |
| **Complexity** | Medium-High |
| **Dependencies** | Device mesh, peer access, synchronization primitives |

**Table Stakes?** YES — Essential for gradient synchronization in deep learning.

**API Surface:**
```cpp
namespace cuda::primitives {

enum class ReduceOp { Sum, Min, Max, Prod };

template<typename T>
class MultiGPUReduce {
public:
    MultiGPUReduce(ncclComm_t* comms, int num_gpus);
    
    // All-reduce: every GPU gets the final reduced value
    void all_reduce(
        const T* send_buffer,
        T* recv_buffer,
        size_t count,
        ReduceOp op,
        cudaStream_t* streams
    );
    
    // Reduce-scatter: result partitioned, each GPU gets its portion
    void reduce_scatter(
        const T* send_buffer,
        T* recv_buffer,
        size_t count_per_gpu,  // count elements in each partition
        ReduceOp op,
        cudaStream_t* streams
    );
    
private:
    ncclComm_t* comms_;
    int num_gpus_;
};

// Convenience: single-stream API wrapping group semantics
template<typename T>
void all_reduce(
    T* buffer,
    size_t count,
    ReduceOp op,
    ncclComm_t comm,
    cudaStream_t stream
);

}
```

#### 3.2 Multi-GPU Broadcast

| Aspect | Details |
|--------|---------|
| **What it does** | Copy data from one GPU to all other GPUs |
| **Complexity** | Medium |
| **Dependencies** | Device mesh, synchronization primitives |

**Table Stakes?** YES — Required for weight synchronization in data parallel training.

**API Surface:**
```cpp
namespace cuda::primitives {

template<typename T>
void broadcast(
    const T* send_buffer,    // valid only on root_gpu
    T* recv_buffer,          // output on all GPUs
    size_t count,
    int root_gpu,
    ncclComm_t comm,
    cudaStream_t stream
);

}
```

#### 3.3 Multi-GPU All-Gather

| Aspect | Details |
|--------|---------|
| **What it does** | Gather data from all GPUs, distribute combined result to all |
| **Use case** | Gradient gathering, embedding table lookup results |
| **Complexity** | Medium |
| **Dependencies** | Device mesh |

**Table Stakes?** YES — Required for tensor parallel and data parallel patterns.

**API Surface:**
```cpp
namespace cuda::primitives {

template<typename T>
void all_gather(
    const T* send_buffer,
    T* recv_buffer,          // must be count * num_gpus elements
    size_t count_per_gpu,
    ncclComm_t comm,
    cudaStream_t stream
);

}
```

#### 3.4 Distributed Batch Normalization

| Aspect | Details |
|--------|---------|
| **What it does** | Batch norm computation spanning multiple GPUs with proper synchronization |
| **Options** | Sync BN (all-reduce mean/variance), local BN (approximation) |
| **Complexity** | High |
| **Dependencies** | All-gather, reduce, synchronization |

**Table Stakes?** NO — Primary use case is deep learning; v1.1 focuses on primitives.
**Differentiator?** YES — "Distributed batch norm" is a strong differentiator for ML workloads.

**Recommendation:** Defer to v1.2 unless explicit user demand.

#### 3.5 Synchronization Primitives (Barriers)

| Aspect | Details |
|--------|---------|
| **What it does** | Block until all GPUs reach a synchronization point |
| **CUDA API** | `cudaDeviceSynchronize()` per device + host-side coordination |
| **NCCL** | `ncclGroupStart()`/`ncclGroupEnd()` provides implicit barrier; explicit via `ncclBarrier()` |
| **Complexity** | Low |
| **Dependencies** | Device mesh |

**Table Stakes?** YES — Required for correctness in multi-GPU programs.

**API Surface:**
```cpp
namespace cuda::primitives {

// Host-side: synchronize all GPU streams
void synchronize_mesh(const std::vector<cudaStream_t>& streams);

// NCCL barrier across communicators
void barrier(ncclComm_t comm, cudaStream_t stream);

// Host-side multi-stream barrier
class MeshBarrier {
public:
    explicit MeshBarrier(int num_devices);
    
    // Signal arrival from one device
    void arrive(cudaStream_t stream);
    
    // Wait for all devices
    void wait();
    
private:
    int num_devices_;
    std::atomic<int> arrived_;
    // ... synchronization primitives
};

}
```

---

### Category 4: Multi-GPU Linear Algebra

**Goal:** Enable large-matrix operations that exceed single-GPU memory/compute capacity.

#### 4.1 Multi-GPU Matrix Multiply

| Aspect | Details |
|--------|---------|
| **What it does** | Distribute matmul computation across GPUs |
| **Approaches** | See comparison below |
| **Complexity** | Very High |
| **Dependencies** | All categories above; distributed memory pool |

**Table Stakes?** YES — Core v1.1 deliverable per PROJECT.md.

#### Approach Comparison

| Approach | Description | Pros | Cons | Best For |
|----------|-------------|------|------|----------|
| **Tensor Parallelism** | Split weight matrix by columns; all-reduce after local matmul | Scales large layers; good for wide layers | High communication; complex implementation | Large transformer layers |
| **Pipeline Parallelism** | Split computation by layers (depth); micro-batches pipeline through | Good for deep models; overlaps communication | Pipeline bubbles; latency | Very deep networks |
| **Data Parallelism** | Each GPU has full model; sync gradients via all-reduce | Simple; near-linear scaling | Memory per GPU = full model size | Small models on many GPUs |
| **Row-Wise Split** | Split output rows across GPUs | Simple communication (all-gather) | Output must fit in single GPU | Wide output matrices |
| **Column-Wise Split** | Split weight matrix columns across GPUs | Input replicated | All-reduce on output | Large weight matrices |

**Recommendation for v1.1:** Start with **row-wise split** for simplicity and add all-gather. This provides:
- Clear mental model (each GPU computes `m/ngpus` rows)
- Communication pattern is straightforward (all-gather on output)
- Builds on existing matmul infrastructure

**API Surface (Row-Wise Split):**
```cpp
namespace cuda::neural {

struct MultiGPUMatmulOptions {
    std::vector<cublasHandle_t> handles;  // one per GPU
    ncclComm_t* comms;                     // NCCL communicators
    float alpha = 1.0f;
    float beta = 0.0f;
    bool trans_a = false;
    bool trans_b = false;
};

// Row-wise split: each GPU computes rows [start, end) of output
void matmul_row_split(
    const float* A,        // Full input matrix [m, k]
    const float* B,        // Full weight matrix [k, n]
    float* C,              // Output distributed across GPUs
    int m, int n, int k,   // Full dimensions
    int start_row,         // Which rows this GPU computes
    int num_rows,          // How many rows this GPU computes
    MultiGPUMatmulOptions options
);

// Convenience: automatic row distribution
void matmul_distributed(
    const float* A,        // Full input [m, k]
    const float* B,        // Full weights [k, n]
    float* C,              // Full output [m, n]
    int m, int n, int k,
    MultiGPUMatmulOptions options
);

}
```

**Future Phases:** Add tensor parallel support for very large individual layers (v1.2+).

#### 4.2 API Design for Multi-GPU Operations

| Principle | Implementation |
|-----------|----------------|
| **Fail gracefully** | Check peer access, fall back to host-mediated copies if unavailable |
| **Stream integration** | All operations accept `cudaStream_t` for async execution |
| **Device context** | RAII guards (`cudaSetDevice` in constructor) |
| **Backwards compatibility** | Single-GPU API unchanged; multi-GPU is additive |

**Wrapper Pattern:**
```cpp
// Existing single-GPU API (unchanged)
void matmul(const float* A, const float* B, float* C, int m, int n, int k, MatmulOptions opts);

// New multi-GPU API (additive)
namespace cuda::neural {

// Distribute across available GPUs automatically
void matmul_distributed(const DistributedTensor& A, const DistributedTensor& B, 
                        DistributedTensor& C, MultiGPUMatmulOptions opts);

// Or explicit device assignment
void matmul_on(const std::vector<int>& devices, ...);

}
```

---

## Feature Dependencies Map

```
GPU Enumeration (1.1)
    │
    ├── Peer Access Queries (1.2)
    │       │
    │       ├── Device Mesh Topology (1.3)
    │       │
    │       └── Async Peer Copy (2.4)
    │               │
    └── Distributed Memory Pool (2.2)
            │
            ├── Multi-GPU Reduce (3.1) ←── Sync Primitives (3.5)
            │
            ├── Multi-GPU Broadcast (3.2) ←── Sync Primitives (3.5)
            │
            ├── Multi-GPU All-Gather (3.3) ←── Sync Primitives (3.5)
            │
            └── Multi-GPU Matmul (4.1) ←── All of the above
```

---

## MVP Recommendation

For v1.1, implement in this order:

### Phase 1: Device Mesh Foundation
1. GPU enumeration and device info queries
2. Peer access capability detection
3. Async peer-to-peer copy primitives

### Phase 2: Collective Primitives (via NCCL wrapper)
4. NCCL communicator setup and management
5. All-reduce (essential for gradient sync)
6. Broadcast
7. All-gather
8. Synchronization barriers

### Phase 3: Multi-GPU Matmul
9. Row-wise split matmul
10. Integration with existing matmul API
11. Tests with distributed computation

### Defer to v1.2+:
- Device mesh topology optimization (Phase 1.3)
- Distributed batch normalization (Phase 3.4)
- Tensor parallelism for large layers (Phase 4.x)
- CUDA MPS multi-process support

---

## Sources

| Source | Confidence | What It Provides |
|--------|------------|------------------|
| [CUDA Programming Guide: Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system) | HIGH | Device enumeration, peer access, P2P copy APIs |
| [NCCL 2.30 Documentation: Collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | HIGH | AllReduce, Broadcast, AllGather semantics and APIs |
| [NCCL 2.30 Documentation: Examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html) | HIGH | Multi-GPU patterns, communicator setup, group semantics |
| [CUDA Programming Guide: Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator) | HIGH | cudaMallocAsync, memory pools, IPC memory pool support |
| Existing Nova codebase (`include/cuda/memory/memory_pool.h`, `include/cuda/neural/matmul.h`) | HIGH | Current API patterns, class structures |

---

## Open Questions

1. **NCCL dependency:** Should v1.1 use NCCL directly or implement custom collectives for educational purposes first? NCCL is the industry standard; custom implementation adds maintenance burden.

2. **Tensor parallelism:** When to add support for models that exceed single-GPU memory? This requires careful design to avoid API churn.

3. **Device selection strategy:** Should the library auto-select optimal GPU placement or require explicit device assignment? Auto-placement is easier; explicit is more flexible.

4. **Error recovery:** How to handle GPU failures in a multi-GPU operation? NCCL has fault tolerance features but they add complexity.
