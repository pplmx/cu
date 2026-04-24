# Architecture Research: NCCL, Tensor & Pipeline Parallelism

**Domain:** C++ CUDA library multi-GPU training infrastructure
**Project:** Nova v1.3 - NCCL Integration, Tensor Parallelism, Pipeline Parallelism
**Researched:** 2026-04-24
**Confidence:** HIGH (NCCL, TP) / MEDIUM (Pipeline scheduling nuances)

## Executive Summary

This architecture research covers three major enhancements to the Nova CUDA library's multi-GPU capabilities:

1. **NCCL Integration** — Replace P2P ring-allreduce fallback with optimized NCCL collectives
2. **Tensor Parallelism** — Column-wise and row-wise matrix splits for large layer support
3. **Pipeline Parallelism** — Micro-batch scheduling for deep model parallelism

Key architectural decisions:
- **NCCL as optional backend** — P2P fallback remains for systems without NCCL
- **Tensor parallelism in distributed layer** — Extends `DistributedMatmul` with TP strategy
- **Pipeline parallelism as scheduling layer** — Separates scheduling logic from computation
- **Backward compatibility preserved** — All existing single-GPU and data-parallel code unchanged

---

## 1. NCCL Integration Architecture

### 1.1 Where NCCL Fits in the Five-Layer Architecture

```
Layer 4: Distributed Operations (EXTENDED)
├── NcclCommContext (NEW) — NCCL communicator management
├── DistributedReduce (EXTENDED) — Add NCCL backend
├── DistributedBroadcast (EXTENDED) — Add NCCL backend
└── DistributedAllGather (EXTENDED) — Add NCCL backend

Layer 2.5: Peer Transport (UNCHANGED)
├── PeerCopy — Still used for non-collective P2P
└── MeshBarrier — Event-based synchronization
```

**Decision:** NCCL integrates at Layer 4 as a backend for collective operations. It does NOT replace the peer transport layer (Layer 2.5) because:
- NCCL handles collectives internally via its own transport (NVLink, PCIe)
- Non-collective P2P operations (e.g., tensor movement in pipeline parallelism) still use `PeerCopy`
- Keeping Layer 2.5 allows P2P fallback when NCCL is unavailable

### 1.2 Singleton vs Dependency-Injected NCCL Context

**Recommendation: Dependency injection via `NcclContext` parameter**

```cpp
// include/cuda/distributed/nccl_context.h
namespace cuda::distributed {

/**
 * @class NcclContext
 * @brief NCCL communicator pool with lazy initialization
 *
 * Manages per-device NCCL communicators and provides thread-safe access.
 * Uses dependency injection pattern — pass NcclContext to operations.
 *
 * @example
 * @code
 * NcclContext ctx;
 * ctx.initialize();  // Creates communicators for all mesh devices
 *
 * NcclReduce reduce(ctx);
 * reduce.all_reduce(data, count, ReductionOp::Sum, dtype);
 * @endcode
 */
class NcclContext {
public:
    struct Config {
        int device_count = 0;
        bool nccl_unique_id_from_env = false;
        std::string nccl_debug_file = "";
    };

    NcclContext() = default;
    explicit NcclContext(const Config& config);
    ~NcclContext();

    // Non-copyable, movable
    NcclContext(const NcclContext&) = delete;
    NcclContext& operator=(const NcclContext&) = delete;
    NcclContext(NcclContext&&) noexcept;
    NcclContext& operator=(NcclContext&&) noexcept;

    /**
     * @brief Initialize NCCL communicators for all devices
     * @param config Configuration options
     */
    void initialize(const Config& config);

    /**
     * @brief Get NCCL communicator for a specific device
     * @param device Device index
     * @return NCCL communicator handle
     */
    ncclComm_t get_comm(int device) const;

    /**
     * @brief Get CUDA stream for a specific device
     * @param device Device index
     * @return CUDA stream
     */
    cudaStream_t get_stream(int device) const;

    /**
     * @brief Check if NCCL is initialized
     * @return true if initialized
     */
    bool initialized() const { return initialized_; }

    /**
     * @brief Get device count
     * @return Number of devices in NCCL group
     */
    int device_count() const { return device_count_; }

    /**
     * @brief Generate NCCL unique ID for multi-process launch
     * @return NCCL unique ID
     */
    static ncclUniqueId generate_unique_id();

private:
    void cleanup();

    int device_count_ = 0;
    std::vector<ncclComm_t> communicators_;
    std::vector<cudaStream_t> streams_;
    bool initialized_ = false;
};

/**
 * @class NcclReduce
 * @brief NCCL-based all-reduce with P2P fallback
 */
class NcclReduce {
public:
    explicit NcclReduce(NcclContext& ctx);
    ~NcclReduce();

    void all_reduce(
        const void* send_data,
        void* recv_data,
        size_t count,
        ReductionOp op,
        ncclDataType_t dtype);

    void all_reduce_async(
        const void* send_data,
        void* recv_data,
        size_t count,
        ReductionOp op,
        cudaStream_t stream,
        ncclDataType_t dtype);

private:
    NcclContext& ctx_;
};

}  // namespace cuda::distributed
```

**Why dependency injection over singleton:**

| Pattern | Pros | Cons |
|---------|------|------|
| Singleton | Simple access, global state | Hard to test, cannot mock, lifetime issues |
| Dependency Injection | Testable, explicit dependencies, composable | More boilerplate |

**Best compromise:** Provide a `NcclContext::instance()` for convenience that returns a static global context, but allow injection for testing.

```cpp
// Convenience global context
NcclContext& global_nccl_context() {
    static NcclContext ctx;
    static bool initialized = false;
    if (!initialized) {
        ctx.initialize({});
        initialized = true;
    }
    return ctx;
}
```

### 1.3 Stream-Based vs Blocking Collective Calls

**Recommendation: Async collectives with explicit streams**

NCCL supports two modes:

1. **Blocking (implicit stream):** `ncclAllReduce(send, recv, count, dtype, op, comm, nullptr)`
2. **Async (explicit stream):** `ncclAllReduce(send, recv, count, dtype, op, comm, stream)`

```cpp
// Async with CUDA stream (RECOMMENDED)
ncclAllReduce(send_data, recv_data, count, dtype, op, comm, cuda_stream);

// Blocking variant — blocks GPU
ncclAllReduce(send_data, recv_data, count, dtype, op, comm, cudaStreamNull);
```

**Integration with existing StreamManager:**

```cpp
// extend distributed/common.h
namespace cuda::distributed {

class MeshStreams {
public:
    // ... existing methods ...

    /**
     * @brief Get NCCL communicator for a device
     * @param device Device index
     * @return NCCL communicator (from NcclContext if initialized)
     */
    ncclComm_t get_nccl_comm(int device);

    /**
     * @brief Check if NCCL is available
     * @return true if NCCL context is initialized
     */
    bool has_nccl() const { return nccl_ctx_ != nullptr; }

private:
    // ... existing members ...
    NcclContext* nccl_ctx_ = nullptr;
};

}  // namespace cuda::distributed
```

### 1.4 NCCL Memory Allocation

NCCL buffers must be:
- **GPU-accessible** — All buffers must be on GPU memory
- **Pinned (optional)** — Improves performance but not required
- **Aligned** — NCCL performs better with aligned buffers

**Integration with MemoryPool:**

```cpp
// Allocate GPU memory via pool, then use with NCCL
auto* buffer = static_cast<float*>(
    distributed_pool.allocate(size * sizeof(float), device_id));

// Use with NCCL (device pointers are valid)
ncclAllReduce(buffer, buffer, size, ncclFloat32, ncclSum, comm, stream);
```

---

## 2. Tensor Parallelism Architecture

### 2.1 Column-Wise and Row-Wise Matrix Splits

Tensor parallelism splits a weight matrix across GPUs. For a matrix multiplication Y = XA:

**Column Parallel (A split along columns):**
```
A = [A_1 | A_2 | ... | A_tp]  // Split columns across TP GPUs
Y_i = X @ A_i                 // Each GPU computes partial result
Y = AllGather(Y_1, Y_2, ...)  // Gather to get full Y
```

**Row Parallel (A split along rows):**
```
X = [X_1 | X_2 | ... | X_tp]  // Split X across GPUs
Y_i = X_i @ A_i               // Each GPU computes partial
Y = AllReduce(Y_1, Y_2, ...)  // Sum partial results
```

### 2.2 Communication Patterns

| Strategy | Input Split | Weight Split | Output Communication |
|----------|-------------|--------------|---------------------|
| Column Parallel | Replicated | Column-wise | All-Gather after matmul |
| Row Parallel | Row-wise | Replicated | All-Reduce after matmul |
| Column + Row (2D) | Row-wise | Column-wise | All-Reduce + All-Gather |

**For transformer layers:**
- **Attention QKV projection:** Column parallel (splits weight, gathers output)
- **Output projection:** Row parallel (scatters input, reduces output)
- **FFN first layer:** Column parallel
- **FFN second layer:** Row parallel

### 2.3 Integration with Existing Matmul

**Extend DistributedMatmul to support tensor parallelism:**

```cpp
// include/cuda/distributed/matmul.h (EXTENDED)
namespace cuda::distributed {

/**
 * @enum ParallelismStrategy
 * @brief Strategy for distributing computation across GPUs
 */
enum class ParallelismStrategy {
    DataParallel,     // Row-partition input (existing v1.1)
    TensorParallelCol, // Column-partition weights (NEW)
    TensorParallelRow  // Row-partition weights (NEW)
};

/**
 * @struct TensorParallelConfig
 * @brief Configuration for tensor parallelism
 */
struct TensorParallelConfig {
    /** Tensor parallelism degree (number of GPUs) */
    int tp_size = 1;

    /** My rank within the TP group */
    int tp_rank = 0;

    /** Total number of TP groups */
    int num_tp_groups = 1;

    /** Enable sequence parallelism (for TP in attention) */
    bool sequence_parallel = false;
};

/**
 * @class TensorParallelMatmul
 * @brief Tensor-parallel matrix multiply operations
 *
 * Implements column-parallel and row-parallel matmul with
 * integrated NCCL collectives for gradient synchronization.
 *
 * @example
 * @code
 * TensorParallelConfig config;
 * config.tp_size = 4;
 * config.tp_rank = local_rank;
 * config.num_tp_groups = num_gpus / 4;
 *
 * TensorParallelMatmul tp_matmul(ctx, config);
 * tp_matmul.column_parallel(
 *     input, weight_part, output_part, m, n, k);
 * // After call: output_part contains portion of full result
 * @endcode
 */
class TensorParallelMatmul {
public:
    TensorParallelMatmul(
        NcclContext& ctx,
        const TensorParallelConfig& config);

    /**
     * @brief Column-parallel matmul: Y = X @ A_split
     *
     * Weight matrix A is split column-wise across TP GPUs.
     * Each GPU computes Y_i = X @ A_i
     * Outputs must be gathered for full result.
     *
     * @param X Input matrix [m x k] (replicated across GPUs)
     * @param A_partial Weight partition [k x (n/tp_size)] on local GPU
     * @param Y_partial Output partition [m x (n/tp_size)]
     * @param m Rows in X and Y
     * @param n Columns in A (global), n/tp_size local
     * @param k Columns in X, rows in A
     */
    void column_parallel(
        const float* X,
        const float* A_partial,
        float* Y_partial,
        int m, int n, int k);

    /**
     * @brief Row-parallel matmul: Y_split = X_split @ A
     *
     * Input X is split row-wise across TP GPUs.
     * Each GPU computes Y_i = X_i @ A
     * Results must be reduced for full result.
     *
     * @param X_partial Input partition [(m/tp_size) x k]
     * @param A Weight matrix [k x n] (replicated across GPUs)
     * @param Y_partial Output partition [(m/tp_size) x n]
     * @param m Rows in X (global), m/tp_size local
     * @param n Columns in A and Y
     * @param k Columns in X, rows in A
     */
    void row_parallel(
        const float* X_partial,
        const float* A,
        float* Y_partial,
        int m, int n, int k);

    /**
     * @brief All-gather for column-parallel output
     *
     * Gathers Y_partial from all TP GPUs to form full Y.
     *
     * @param Y_partial Local output partition
     * @param Y_full Full output buffer (size m * n)
     * @param m Rows
     * @param n Columns
     */
    void gather_output(
        const float* Y_partial,
        float* Y_full,
        int m, int n);

    /**
     * @brief All-reduce for row-parallel output
     *
     * Reduces Y_partial from all TP GPUs.
     *
     * @param Y_partial Local output partition (modified in-place)
     * @param m Rows
     * @param n Columns
     */
    void reduce_output(float* Y_partial, int m, int n);

private:
    NcclContext& ctx_;
    TensorParallelConfig config_;
    std::unique_ptr<NcclAllGather> all_gather_;
    std::unique_ptr<NcclAllReduce> all_reduce_;
};

}  // namespace cuda::distributed
```

### 2.4 Data Flow: Transformer Layer with Tensor Parallelism

```
Transformer Layer with TP=4:

Input X (replicated)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ QKV Projection (Column Parallel)                           │
│   Q = X @ W_q_split  ┐                                     │
│   K = X @ W_k_split  ├─► [AllGather Q, K, V]               │
│   V = X @ W_v_split  │   (each GPU gets full Q, K, V)      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Attention Score (local)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Output Projection (Row Parallel)                           │
│   Y_partial = AttnScore @ W_o_split                        │
│   Y = [AllReduce Y_partial]                                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ FFN (Column Parallel → Row Parallel)                       │
│   Hidden1 = X @ W1_split  ┌ (Column parallel, no comm)    │
│   Hidden2 = GELU(Hidden1) │                               │
│   Output = Hidden2 @ W2   │ (Row parallel)                │
│   Output = [AllReduce Output]                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Parallelism Architecture

### 3.1 Micro-Batching Strategies

Pipeline parallelism splits model layers across GPUs (vertical split). Micro-batching enables overlapping forward and backward passes.

**Key concepts:**
- **Micro-batch:** A small portion of the global batch, processed independently
- **Pipeline stage:** A group of layers assigned to one GPU
- **Pipeline schedule:** Order of forward/backward passes across microbatches

### 3.2 Forward/Backward Pass Overlapping

**1F1B Schedule (One Forward, One Backward):**

```
Time →
GPU0: [F0][F1][F2][F3][B3][B2][B1][B0]
GPU1:     [F0][F1][F2][F3][B3][B2][B1][B0]
GPU2:         [F0][F1][F2][F3][B3][B2][B1][B0]
GPU3:             [F0][F1][F2][F3][B3][B2][B1][B0]
```

**Interleaved 1F1B (better load balancing):**

```
Time →
GPU0: [F0][F1][B1][F0][F1][B1][F0][F1][B1]...
GPU1:     [F2][F3][B3][F2][F3][B3][F2][F3][B3]...
```

### 3.3 Communication Scheduling

Pipeline parallelism requires P2P communication between stages:
- **Forward:** Send activation to next stage
- **Backward:** Send gradients to previous stage

```cpp
// include/cuda/pipeline/pipeline_scheduler.h
namespace cuda::pipeline {

/**
 * @struct PipelineConfig
 * @brief Configuration for pipeline parallelism
 */
struct PipelineConfig {
    /** Number of pipeline stages (GPUs) */
    int num_stages = 1;

    /** My stage rank (0 = first stage, num_stages-1 = last) */
    int stage_rank = 0;

    /** Number of microbatches in global batch */
    int num_microbatches = 1;

    /** Enable interleaved schedule */
    bool interleaved = false;

    /** Number of micro-batches per stage per iteration */
    int num_microbatches_per_stage = 1;
};

/**
 * @class PipelineScheduler
 * @brief Schedules forward/backward passes for pipeline parallelism
 *
 * Manages micro-batch execution, P2P communication, and gradient
 * synchronization across pipeline stages.
 *
 * @example
 * @code
 * PipelineConfig config;
 * config.num_stages = 4;
 * config.stage_rank = my_rank;
 * config.num_microbatches = 16;
 *
 * PipelineScheduler scheduler(ctx, config);
 *
 * for (int iter = 0; iter < num_iterations; ++iter) {
 *     scheduler.run_pipeline(
 *         forward_fn, backward_fn,
 *         input, labels);
 * }
 * @endcode
 */
class PipelineScheduler {
public:
    using ForwardFn = std::function<void(const Tensor&, Tensor*)>;
    using BackwardFn = std::function<void(const Tensor&, const Tensor&, Tensor*)>;

    PipelineScheduler(
        NcclContext& ctx,
        const PipelineConfig& config);

    ~PipelineScheduler();

    /**
     * @brief Run one pipeline iteration
     *
     * Executes forward and backward passes for all microbatches
     * according to the configured schedule.
     *
     * @param forward_fn Function to compute forward pass
     * @param backward_fn Function to compute backward pass
     * @param input Input tensor for first stage
     * @param labels Labels for loss computation (last stage only)
     */
    void run_pipeline(
        ForwardFn forward_fn,
        BackwardFn backward_fn,
        Tensor& input,
        const Tensor& labels);

    /**
     * @brief Run 1F1B schedule
     */
    void run_1f1b(
        ForwardFn forward_fn,
        BackwardFn backward_fn);

    /**
     * @brief Run interleaved 1F1B schedule
     */
    void run_interleaved_1f1b(
        ForwardFn forward_fn,
        BackwardFn backward_fn);

    /**
     * @brief Receive activation from previous stage
     * @param target Buffer to receive into
     */
    void recv_forward(Tensor* target);

    /**
     * @brief Send activation to next stage
     * @param source Activation to send
     */
    void send_forward(const Tensor& source);

    /**
     * @brief Receive gradient from next stage
     * @param target Buffer to receive into
     */
    void recv_backward(Tensor* target);

    /**
     * @brief Send gradient to previous stage
     * @param source Gradient to send
     */
    void send_backward(const Tensor& source);

private:
    NcclContext& ctx_;
    PipelineConfig config_;

    // P2P communication
    std::vector<cuda::mesh::PeerCopy> peer_copies_;
    std::vector<cudaEvent_t> events_;

    // Pipeline buffers (ping-pong for overlap)
    std::vector<Tensor> forward_buffers_;
    std::vector<Tensor> backward_buffers_;
    int buffer_idx_ = 0;
};

}  // namespace cuda::pipeline
```

### 3.4 Integration with Existing Components

```
┌─────────────────────────────────────────────────────────────┐
│ cuda::pipeline::PipelineScheduler                          │
│ - Manages micro-batch scheduling                           │
│ - Calls forward_fn/backward_fn                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ cuda::distributed::TensorParallelMatmul                    │
│ - Computes layer operations within each stage              │
│ - Uses NCCL for gradient synchronization                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ cuda::distributed::NcclContext                             │
│ - Provides NCCL communicators and streams                  │
│ - Handles P2P communication for pipeline stages            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Integration Points

### 4.1 Extend Existing DeviceMesh for NCCL Topology

```cpp
// Extend device_mesh.h
namespace cuda::mesh {

class DeviceMesh {
public:
    // ... existing methods ...

    /**
     * @brief Get NCCL unique ID for multi-process initialization
     * @return NCCL unique ID
     */
    ncclUniqueId get_nccl_unique_id();

    /**
     * @brief Get local rank within node
     * @return Local rank (0 for single-node)
     */
    int local_rank() const;

    /**
     * @brief Get world size (total GPUs in training)
     * @return Total number of GPUs
     */
    int world_size() const { return device_count_; }

    /**
     * @brief Check if NCCL is available
     * @return true if NCCL can be initialized
     */
    bool has_nccl_support() const;
};

}  // namespace cuda::mesh
```

### 4.2 Hook into Existing Memory Pool for Tensor-Allocated Buffers

```cpp
// extend distributed_pool.h
namespace cuda::memory {

class DistributedMemoryPool {
public:
    // ... existing methods ...

    /**
     * @brief Allocate tensor-aligned buffer for NCCL
     *
     * Allocates GPU memory with alignment suitable for NCCL operations.
     *
     * @param bytes Size in bytes
     * @param device Device to allocate on
     * @param alignment Alignment requirement (default: 4096)
     * @return Pointer to allocated memory
     */
    void* allocate_tensor(size_t bytes, int device, size_t alignment = 4096);

    /**
     * @brief Allocate buffer for tensor parallelism
     *
     * Allocates buffer sized for tensor-parallel portion.
     *
     * @param bytes_per_gpu Bytes needed on each GPU
     * @param device Device to allocate on
     * @return Pointer to allocated memory
     */
    void* allocate_tensor_parallel(size_t bytes_per_gpu, int device);
};

}  // namespace cuda::memory
```

### 4.3 Leverage Existing Multi-GPU Matmul Infrastructure

```cpp
// New file: include/cuda/distributed/tensor_parallel_matmul.h

// DistributedMatmul already uses neural::matmul per device
// TensorParallelMatmul extends this pattern:

class TensorParallelMatmul {
private:
    // Reuse existing per-device cublas handles
    std::vector<cublasHandle_t> cublas_handles_;

    // Reuse neural matmul implementation
    void local_matmul(
        const float* A,
        const float* B,
        float* C,
        int m, int n, int k);
};
```

---

## 5. Project Structure for v1.3

```
include/cuda/
    nccl/                              # NEW: NCCL integration
    │   ├── nccl_context.h            # NcclContext, communicator pool
    │   ├── nccl_collective.h         # Base class for NCCL collectives
    │   ├── nccl_reduce.h             # NCCL all-reduce
    │   ├── nccl_all_gather.h         # NCCL all-gather
    │   ├── nccl_broadcast.h          # NCCL broadcast
    │   └── nccl_reduce_scatter.h     # NCCL reduce-scatter
    │
    tensor_parallel/                   # NEW: Tensor parallelism
    │   ├── tensor_parallel_matmul.h  # Column/row parallel matmul
    │   ├── column_parallel_layer.h   # Column-parallel linear layer
    │   ├── row_parallel_layer.h      # Row-parallel linear layer
    │   └── tensor_parallel_config.h  # Configuration
    │
    pipeline/                          # NEW: Pipeline parallelism
    │   ├── pipeline_scheduler.h      # 1F1B, interleaved schedules
    │   ├── p2p_communication.h       # Send/recv primitives
    │   └── pipeline_config.h         # Configuration
    │
    distributed/                       # EXISTING: Extended
    │   ├── matmul.h                  # EXTENDED: Add TP strategies
    │   ├── reduce.h                  # EXTENDED: Add NCCL backend
    │   ├── all_gather.h              # EXTENDED: Add NCCL backend
    │   ├── broadcast.h               # EXTENDED: Add NCCL backend
    │   └── barrier.h                 # EXTENDED: Add NCCL backend

src/cuda/
    nccl/                              # NEW: NCCL implementation
    │   ├── nccl_context.cpp
    │   ├── nccl_reduce.cpp
    │   ├── nccl_all_gather.cpp
    │   └── ...
    │
    tensor_parallel/                   # NEW: TP implementation
    │   ├── tensor_parallel_matmul.cpp
    │   ├── column_parallel_layer.cpp
    │   └── row_parallel_layer.cpp
    │
    pipeline/                          # NEW: Pipeline implementation
    │   ├── pipeline_scheduler.cpp
    │   └── p2p_communication.cpp
```

---

## 6. Build System Changes

```cmake
# NCCL support (optional)
option(NOVA_ENABLE_NCCL "Enable NCCL for optimized multi-GPU collectives" ON)

if(NOVA_ENABLE_NCCL)
    find_package(NCCL 2.18 REQUIRED)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES nccl)
endif()

# New directory: NCCL
add_library(cuda_nccl STATIC
    ${CMAKE_SOURCE_DIR}/src/cuda/nccl/nccl_context.cpp
    ${CMAKE_SOURCE_DIR}/src/cuda/nccl/nccl_reduce.cpp
    ${CMAKE_SOURCE_DIR}/src/cuda/nccl/nccl_all_gather.cpp
    ${CMAKE_SOURCE_DIR}/src/cuda/nccl/nccl_broadcast.cpp
)
target_include_directories(cuda_nccl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cuda/nccl>
)
target_link_libraries(cuda_nccl PUBLIC
    cuda_impl
    CUDA::cudart
)
if(NCCL_FOUND)
    target_link_libraries(cuda_nccl PUBLIC NCCL::nccl)
    target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=1)
else()
    target_compile_definitions(cuda_nccl PUBLIC NOVA_NCCL_ENABLED=0)
endif()

# New directory: Tensor Parallelism
add_library(cuda_tensor_parallel STATIC
    ${CMAKE_SOURCE_DIR}/src/cuda/tensor_parallel/tensor_parallel_matmul.cpp
    ${CMAKE_SOURCE_DIR}/src/cuda/tensor_parallel/column_parallel_layer.cpp
    ${CMAKE_SOURCE_DIR}/src/cuda/tensor_parallel/row_parallel_layer.cpp
)
target_include_directories(cuda_tensor_parallel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cuda/tensor_parallel>
)
target_link_libraries(cuda_tensor_parallel PUBLIC
    cuda_nccl
    cuda_neural
)

# New directory: Pipeline Parallelism
add_library(cuda_pipeline STATIC
    ${CMAKE_SOURCE_DIR}/src/cuda/pipeline/pipeline_scheduler.cpp
    ${CMAKE_SOURCE_DIR}/src/cuda/pipeline/p2p_communication.cpp
)
target_include_directories(cuda_pipeline PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/cuda/pipeline>
)
target_link_libraries(cuda_pipeline PUBLIC
    cuda_tensor_parallel
    cuda_mesh
)

# Update cuda_multigpu to include new components
set_target_properties(cuda_multigpu PROPERTIES
    INTERFACE_LINK_LIBRARIES
        "cuda_nccl;cuda_tensor_parallel;cuda_pipeline"
)
```

---

## 7. Anti-Patterns

### Anti-Pattern 1: NCCL Blocking Operations in Hot Loops

**Wrong:**
```cpp
// BAD: Blocking on every iteration
while (training) {
    ncclAllReduce(gpu_data, result, size, ncclFloat32, ncclSum, comm, nullptr);
    cudaStreamSynchronize(stream);  // Blocks GPU!
}
```

**Right:**
```cpp
// GOOD: Async operations with proper synchronization
while (training) {
    // Schedule async collective
    CUDA_CHECK(cudaEventRecord(collective_done, stream));
    // Schedule dependent work
    launch_gradient_update(stream);
    // Only sync when needed
    CUDA_CHECK(cudaStreamWaitEvent(weight_update_stream, collective_done));
}
```

### Anti-Pattern 2: Mismatched NCCL/Model Parallelism Degrees

**Wrong:**
```cpp
// Configured for TP=4 but only have 2 GPUs
NcclContext ctx;
ctx.initialize({{.device_count = 2}});
TensorParallelConfig tp_config;
tp_config.tp_size = 4;  // WRONG: Only 2 GPUs!
```

**Right:**
```cpp
// Validate configuration
int num_gpus = DeviceMesh::instance().device_count();
int tp_size = std::min(num_gpus, max_tp_size);
int num_tp_groups = num_gpus / tp_size;
```

### Anti-Pattern 3: Pipeline Imbalance

**Wrong:**
```cpp
// Unequal layer distribution causes bubbles
Stage 0: 50 layers → Stage 1: 2 layers  // Unbalanced!
```

**Right:**
```cpp
// Balanced distribution
int layers_per_stage = total_layers / num_stages;
// Or use Megatron-style virtual pipeline stages
```

### Anti-Pattern 4: Memory Explosions in Pipeline

**Wrong:**
```cpp
// Store all activations for all microbatches
std::vector<Tensor> all_activations;  // Memory explosion!
```

**Right:**
```cpp
// Store only activations for in-flight microbatches
// Use activation checkpointing for memory reduction
int max_inflight = num_microbatches_per_stage;
std::vector<Tensor> activations(max_inflight);
```

---

## 8. Sources

### NCCL
- **NVIDIA NCCL GitHub:** https://github.com/NVIDIA/nccl
- **NCCL Documentation:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- **NCCL Collective Operations:** all-reduce, all-gather, reduce, broadcast, reduce-scatter

### Tensor Parallelism
- **NVIDIA Megatron-LM:** https://github.com/NVIDIA/Megatron-LM
- **Megatron Core Documentation:** https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html
- **Tensor Parallelism Guide:** https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- **ColumnParallelLinear, RowParallelLinear:** `megatron/core/tensor_parallel/layers.py`
- **Tensor Mappings:** `megatron/core/tensor_parallel/mappings.py`

### Pipeline Parallelism
- **Megatron Pipeline Parallelism:** https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/pipeline_parallel.html
- **Pipeline Schedules:** `megatron/core/pipeline_parallel/schedules.py`
- **P2P Communication:** `megatron/core/pipeline_parallel/p2p_communication.py`
- **1F1B Schedule:** One Forward, One Backward per micro-batch
- **Interleaved 1F1B:** Better load balancing with virtual pipeline stages

### DeepSpeed
- **DeepSpeed GitHub:** https://github.com/deepspeedai/DeepSpeed
- **3D Parallelism:** Combines Data + Tensor + Pipeline parallelism
- **Pipeline Schedules:** Similar 1F1B patterns with micro-batching

---

*Architecture research for: Nova CUDA Library v1.3 NCCL/Tensor/Pipeline Parallelism*
*Researched: 2026-04-24*
