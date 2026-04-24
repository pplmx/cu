# Technology Stack: NCCL Integration, Tensor & Pipeline Parallelism

**Project:** Nova CUDA Library v1.2+ — Production NCCL and Parallelism Strategies
**Researched:** 2026-04-24
**Confidence:** HIGH (NCCL docs), MEDIUM (tensor parallelism libraries), MEDIUM (pipeline parallelism)

## Executive Summary

Building on v1.1's foundational multi-GPU support (device mesh, P2P fallback collectives), v1.2 adds production-grade NCCL integration and parallelism strategies for large model support. Key decisions:

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **NCCL** | NCCL 2.25+ | CUDA 20 support, modern API (ncclCommInitRankConfig, ncclCommSplit) |
| **Tensor Parallelism** | Custom implementation | Megatron-LM is too opinionated; implement split strategies directly |
| **Pipeline Parallelism** | GPipe-style with async scheduling | Simpler than PipeDream, better throughput via micro-batching |
| **Avoid** | DeepSpeed, Megatron-LM integration | Over-engineered for single-node, adds large dependencies |

---

## 1. NCCL Library Integration

### Version Compatibility

**Recommended: NCCL 2.25+ (bundled with CUDA 20)**

| NCCL Version | CUDA Support | Key Features for Nova |
|--------------|--------------|----------------------|
| **2.25+** | CUDA 12.0-20 | `ncclCommInitRankConfig`, communicator split/shrink, async error handling |
| 2.22-2.24 | CUDA 11.8-12.x | Good for transition, basic collectives |
| 2.18-2.21 | CUDA 11.x | Minimum viable for modern features |

**CUDA Compatibility Matrix:**

| CUDA Version | NCCL 2.25 | NCCL 2.30 |
|--------------|-----------|-----------|
| CUDA 12.x | Supported | Supported |
| CUDA 17+ | Supported | Supported |
| SM 90 (H100) | Supported | Supported |
| SM 100 (B100/B200) | Supported | Supported |

**Source:** [NCCL 2.30 Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

### NCCL Header Integration

```cpp
// include/cuda/collective/nccl_context.h
#include <nccl.h>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <optional>

namespace cuda::collective {

// NCCL-specific error checking
#define NCCL_CHECK(call) \
  do { \
    ncclResult_t err = call; \
    if (err != ncclSuccess) { \
      throw NcclException( \
        ncclGetErrorString(err), \
        err, \
        #call, \
        __FILE__, \
        __LINE__ \
      ); \
    } \
  } while (0)

// Exception for NCCL errors
class NcclException : public std::runtime_error {
public:
  ncclResult_t error_code;
  const char* expression;
  const char* file;
  int line;
  
  NcclException(const char* msg, ncclResult_t code, 
                const char* expr, const char* f, int l)
    : std::runtime_error(msg), error_code(code), 
      expression(expr), file(f), line(l) {}
};

// NCCL communicator wrapper
class NcclContext {
public:
  // Singleton pattern for single-process multi-GPU
  static NcclContext& instance() {
    static NcclContext ctx;
    return ctx;
  }
  
  // Initialize communicators for specified devices
  void initialize(const std::vector<int>& devices);
  
  // Get communicator for specific device rank
  ncclComm_t comm(int rank) const { 
    return communicators_.at(rank); 
  }
  
  // Get rank count
  int rank_count() const { return static_cast<int>(communicators_.size()); }
  
  // Cleanup - must be called before CUDA context destruction
  void destroy();
  
  ~NcclContext() { destroy(); }

private:
  NcclContext() = default;
  
  std::vector<ncclComm_t> communicators_;
  std::mutex init_mutex_;
  bool initialized_ = false;
};

// Key collective operations wrapper
class NcclCollectives {
public:
  explicit NcclCollectives(ncclComm_t comm) : comm_(comm) {}
  
  // AllReduce: sum across all ranks
  void all_reduce(const void* sendbuff, void* recvbuff, size_t count,
                  ncclDataType_t datatype, ncclRedOp_t op, 
                  cudaStream_t stream) {
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype, 
                             op, comm_, stream));
  }
  
  // Broadcast: send from root to all
  void broadcast(const void* sendbuff, void* recvbuff, size_t count,
                 ncclDataType_t datatype, int root, cudaStream_t stream) {
    NCCL_CHECK(ncclBroadcast(sendbuff, recvbuff, count, datatype,
                             root, comm_, stream));
  }
  
  // AllGather: gather from all, broadcast to all
  void all_gather(const void* sendbuff, void* recvbuff, size_t sendcount,
                  ncclDataType_t datatype, cudaStream_t stream) {
    NCCL_CHECK(ncclAllGather(sendbuff, recvbuff, sendcount, datatype,
                             comm_, stream));
  }
  
  // ReduceScatter: reduce across all, scatter to ranks
  void reduce_scatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                      ncclDataType_t datatype, ncclRedOp_t op, 
                      cudaStream_t stream) {
    NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype,
                                 op, comm_, stream));
  }
  
  // Point-to-point send/recv
  void send(const void* sendbuff, size_t count, ncclDataType_t datatype,
            int peer, cudaStream_t stream) {
    NCCL_CHECK(ncclSend(sendbuff, count, datatype, peer, comm_, stream));
  }
  
  void recv(void* recvbuff, size_t count, ncclDataType_t datatype,
            int peer, cudaStream_t stream) {
    NCCL_CHECK(ncclRecv(recvbuff, count, datatype, peer, comm_, stream));
  }

private:
  ncclComm_t comm_;
};

}  // namespace cuda::collective
```

### Initialization Patterns

**Simple Initialization (Single Process, All GPUs):**

```cpp
// Single-threaded initialization using ncclCommInitAll
void init_simple() {
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  
  std::vector<int> devices(device_count);
  std::iota(devices.begin(), devices.end(), 0);
  
  NcclContext::instance().initialize(devices);
}
```

**Advanced Initialization (With Config):**

```cpp
// Multi-process or custom topology using ncclCommInitRankConfig
void init_advanced(const std::vector<int>& devices) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;  // Non-blocking for async error handling
  config.minCTAs = 1;
  config.maxCTAs = 32;
  config.commName = "nova_collective";
  
  ncclUniqueId unique_id;
  NCCL_CHECK(ncclGetUniqueId(&unique_id));
  
  // In multi-process: broadcast unique_id to all ranks
  // Here: single-process, ranks match device indices
  
  for (int rank = 0; rank < devices.size(); ++rank) {
    CUDA_CHECK(cudaSetDevice(devices[rank]));
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRankConfig(&comm, devices.size(), 
                                      unique_id, rank, &config));
    // Store comm in context
  }
}
```

**Communicator Split (For Sub-Groups):**

```cpp
// Split communicator by "pipeline stage" for pipeline parallelism
ncclComm_t split_by_stage(ncclComm_t parent_comm, int color, int key) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclComm_t new_comm;
  
  NCCL_CHECK(ncclCommSplit(parent_comm, color, key, &new_comm, &config));
  return new_comm;
}
```

### CMake Integration

```cmake
# In CMakeLists.txt
find_package(NCCL 2.25 QUIET)
if(NCCL_FOUND)
  message(STATUS "NCCL ${NCCL_VERSION} found")
  
  # NCCL include path (often in CUDA toolkit)
  set(NCCL_INCLUDE_DIRS 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/nccl
    CACHE PATH "NCCL include directories")
  
  # Create imported target
  add_library(NCCL::nccl SHARED IMPORTED)
  set_target_properties(NCCL::nccl PROPERTIES
    IMPORTED_LOCATION ${NCCL_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}")
  
  set(NOVA_NCCL_ENABLED TRUE)
else()
  message(STATUS "NCCL not found — using P2P fallback")
  set(NOVA_NCCL_ENABLED FALSE)
endif()

# New collective library
add_library(cuda_collective STATIC
  ${CMAKE_SOURCE_DIR}/src/cuda/collective/nccl_ops.cu
  ${CMAKE_SOURCE_DIR}/src/cuda/collective/nccl_context.cpp
)

target_include_directories(cuda_collective PUBLIC
  ${CMAKE_SOURCE_DIR}/include/cuda/collective
)

if(NCCL_FOUND)
  target_link_libraries(cuda_collective PUBLIC 
    NCCL::nccl
    CUDA::cudart
  )
  target_compile_definitions(cuda_collective PUBLIC NOVA_NCCL_ENABLED=1)
else()
  target_compile_definitions(cuda_collective PUBLIC NOVA_NCCL_ENABLED=0)
endif()

set_target_properties(cuda_collective PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
```

### Data Types Mapping

```cpp
// Map CUDA dtype to NCCL dtype
ncclDataType_t to_nccl_dtype(cudaDataType_t cuda_dtype) {
  switch (cuda_dtype) {
    case CUDA_R_8I:   return ncclInt8;
    case CUDA_R_32I:  return ncclInt32;
    case CUDA_R_32F:  return ncclFloat32;
    case CUDA_R_64F:  return ncclFloat64;
    case CUDA_R_16F:  return ncclFloat16;
    case CUDA_R_16BF: return ncclBfloat16;
    // FP8 support (CUDA 11.8+, SM 90+)
    case CUDA_R_8F_E4M3: return ncclFloat8e4m3;
    case CUDA_R_8F_E5M2: return ncclFloat8e5m2;
    default:          return ncclFloat32;
  }
}
```

---

## 2. Tensor Parallelism Implementation

### Strategy Overview

**Recommendation: Custom implementation, NOT Megatron-LM integration**

| Approach | Pros | Cons |
|----------|------|------|
| **Custom Implementation** | Full control, no external dep, matches Nova API | More code to write |
| **Megatron-LM Integration** | Battle-tested, optimized | Opinionated API, large dependency, overkill for single-node |
| **DeepSpeed ZeRO** | Memory efficient | Designed for multi-node, complex integration |

**Why Custom:**
- Nova has existing matmul with cuBLAS handles — wrap, not replace
- Single-node scope means simpler communication patterns (no network)
- Tensor parallelism is additive to existing data parallelism

### Tensor Split Strategies

```cpp
// include/cuda/distributed/tensor_parallel.h

namespace cuda::distributed {

enum class TensorParallelStrategy {
  RowParallel,    // Split along rows (input dimension)
  ColumnParallel, // Split along columns (output dimension)  
  SequenceParallel // Split sequence dimension (for transformers)
};

// Configuration for tensor-parallel matmul
struct TensorParallelConfig {
  int num_stages = 1;  // Number of stages in pipeline parallel
  int num_partitions = 1;  // Number of tensor partitions
  TensorParallelStrategy strategy = TensorParallelStrategy::RowParallel;
  bool reduce_output = true;  // All-reduce final result?
};

}  // namespace cuda::distributed
```

### Column-Parallel Matmul

**Pattern:** Weight matrix W is column-partitioned across GPUs. Each GPU computes partial output columns.

```
GPU 0: W[:, 0:k/2] → Y[:, 0:k/2]
GPU 1: W[:, k/2:k] → Y[:, k/2:k]

Required: All-reduce at output to combine partial results
```

```cpp
// Column-parallel: Y = X @ W, where W is column-partitioned
template<typename T>
void column_parallel_matmul(
    const T* X,           // Full input [m, k]
    const T* W_partial,   // Partitioned weight [k, n/num_gpus]
    T* Y_partial,         // Partial output [m, n/num_gpus]
    int m, int k, int n,
    const TensorParallelConfig& config,
    cudaStream_t stream
) {
  int num_gpus = config.num_partitions;
  int rank = get_current_rank();
  
  // Each GPU does local matmul
  cublasHandle_t handle = get_cublas_handle(rank);
  int local_n = n / num_gpus;
  
  cuda::neural::matmul_impl(
    X, W_partial, Y_partial,
    m, local_n, k,
    handle, stream
  );
  
  // All-reduce to combine partial results
  if (config.reduce_output) {
    std::vector<T*> recv_buffers(num_gpus);
    std::vector<cuda::memory::unique_ptr<T>> temp_buffers;
    
    for (int i = 0; i < num_gpus; ++i) {
      temp_buffers.push_back(cuda::memory::make_unique<T>(m * local_n, rank));
      recv_buffers[i] = temp_buffers[i].get();
    }
    
    NCCL_CHECK(ncclAllReduce(
      Y_partial, recv_buffers[rank],
      m * local_n, to_nccl_dtype<T>(),
      ncclSum, comm_, stream
    ));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpyAsync(
      Y_partial, recv_buffers[rank],
      m * local_n * sizeof(T),
      cudaMemcpyDeviceToDevice, stream
    ));
  }
}
```

### Row-Parallel Matmul

**Pattern:** Input matrix X is row-partitioned. Each GPU computes partial rows of output.

```
GPU 0: X[0:m/2, :] @ W → Y[0:m/2, :]
GPU 1: X[m/2:m, :] @ W → Y[m/2:m, :]

Required: All-gather before computation (each GPU needs full W)
```

```cpp
// Row-parallel: Y = X @ W, where X is row-partitioned
template<typename T>
void row_parallel_matmul(
    const T* X_partial,   // Partitioned input [m/num_gpus, k]
    const T* W,           // Full weight [k, n]
    T* Y_out,             // Output [m, n] (full)
    int m, int k, int n,
    const TensorParallelConfig& config,
    cudaStream_t stream
) {
  int num_gpus = config.num_partitions;
  int rank = get_current_rank();
  int local_m = m / num_gpus;
  
  // All-gather to get full input on each GPU
  std::vector<T> X_full(m * k);
  std::vector<T*> recv_buffers(num_gpus);
  for (int i = 0; i < num_gpus; ++i) {
    recv_buffers[i] = X_full.data() + i * local_m * k;
  }
  
  NCCL_CHECK(ncclAllGather(
    X_partial, recv_buffers[rank],
    local_m * k, to_nccl_dtype<T>(),
    comm_, stream
  ));
  
  // Wait for gather to complete
  cudaStreamSynchronize(stream);
  
  // Local matmul with full X
  cublasHandle_t handle = get_cublas_handle(rank);
  T* local_output = Y_out + rank * local_m * n;
  
  cuda::neural::matmul_impl(
    X_full.data(), W, local_output,
    m, n, k, handle, stream
  );
}
```

### AllReduce Patterns for Tensor Parallelism

```cpp
// include/cuda/distributed/tensor_ops.h

namespace cuda::distributed {

// All-reduce for column-parallel outputs
// Each GPU has [m, n/num_gpus] — combine to full [m, n]
template<typename T>
void all_reduce_column_output(
    T* buffer,        // In/out: [m, n/num_gpus]
    int m, int n,     // Local dimensions
    int num_gpus,
    ncclComm_t comm,
    cudaStream_t stream
) {
  size_t local_count = m * n / num_gpus;
  
  NCCL_CHECK(ncclAllReduce(
    buffer, buffer, local_count,
    to_nccl_dtype<T>(), ncclSum,
    comm, stream
  ));
}

// All-gather for row-parallel inputs
// Each GPU has [m/num_gpus, k] — gather to full [m, k]
template<typename T>
void all_gather_row_input(
    const T* local_input,
    T* full_output,   // [m, k]
    int m, int k,
    int num_gpus,
    int rank,
    ncclComm_t comm,
    cudaStream_t stream
) {
  int local_m = m / num_gpus;
  size_t send_count = local_m * k;
  
  std::vector<const T*> recv_buffers(num_gpus);
  for (int i = 0; i < num_gpus; ++i) {
    recv_buffers[i] = full_output + i * local_m * k;
  }
  
  NCCL_CHECK(ncclAllGather(
    local_input, recv_buffers[rank],
    send_count, to_nccl_dtype<T>(),
    comm, stream
  ));
}

}  // namespace cuda::distributed
```

---

## 3. Pipeline Parallelism Patterns

### Strategy Comparison

| Strategy | Throughput | Memory | Complexity | Best For |
|----------|-----------|--------|------------|----------|
| **GPipe** | High (with bubbles) | Moderate | Low | Simple implementation, large models |
| **PipeDream** | Higher (1F1B) | Higher | Medium | Latency-sensitive training |
| **PipeDream-2BW** | Higher | Lower | High | Memory-constrained scenarios |

**Recommendation: GPipe-style with async micro-batch scheduling**

- Simpler than PipeDream (no weight staleness tracking)
- Effective throughput via large micro-batch sizes
- Compatible with existing NCCL context

### GPipe Implementation

```cpp
// include/cuda/distributed/pipeline.h

namespace cuda::distributed {

struct PipelineConfig {
  int num_stages = 1;           // Number of pipeline stages (GPUs)
  int num_micro_batches = 1;    // Number of micro-batches
  int batch_size = 32;          // Global batch size
  int micro_batch_size = 4;     // Size per micro-batch
};

enum class PipelineStage {
  Forward,
  Backward
};

// Pipeline context manages micro-batch scheduling
class PipelineContext {
public:
  PipelineContext(int num_stages, int num_microbatches);
  
  // Get next micro-batch index and whether to do forward/backward
  // Returns false when pipeline is complete
  bool next_step(int* microbatch_idx, PipelineStage* stage);
  
  // Get which GPU owns this pipeline stage
  int stage_owner(int stage) const { 
    return stage % num_stages_; 
  }
  
  // Check if current rank should execute this step
  bool is_my_turn(int stage, int microbatch) const;
  
private:
  int num_stages_;
  int num_microbatches_;
  int current_step_ = 0;
};

// GPipe barrier for pipeline stages
void pipeline_barrier(int stage, ncclComm_t comm);

// Forward pass for a micro-batch
template<typename T>
void forward_micro_batch(
    const T* input,
    T* output,
    int microbatch_idx,
    int stage,
    cudaStream_t stream
) {
  // Execute layer(s) assigned to this stage
  // For transformer: attention + FFN for middle stages
  execute_stage_forward(input, output, stage, stream);
}

// Backward pass for a micro-batch  
template<typename T>
void backward_micro_batch(
    const T* grad_output,
    T* grad_input,
    T* grad_weights,
    int microbatch_idx,
    int stage,
    cudaStream_t stream
) {
  // Compute gradients through this stage
  execute_stage_backward(grad_output, grad_input, grad_weights, 
                         stage, stream);
}

}  // namespace cuda::distributed
```

### GPipe Scheduling (Interleaved)

```cpp
// src/cuda/distributed/pipeline.cpp

namespace cuda::distributed {

// Interleaved 1F1B scheduling
// Reduces pipeline bubbles by overlapping forward/backward
void run_interleaved_pipeline(
    PipelineContext& ctx,
    const std::vector<int>& devices,
    std::function<void(int, cudaStream_t)>& forward_fn,
    std::function<void(int, cudaStream_t)>& backward_fn
) {
  int num_stages = ctx.num_stages();
  std::vector<cudaStream_t> streams(num_stages);
  
  // Create streams per stage
  for (int i = 0; i < num_stages; ++i) {
    CUDA_CHECK(cudaSetDevice(devices[i]));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }
  
  // Allocate activation/gradient buffers for each stage
  std::vector<std::vector<void*>> activation_bufs(num_stages);
  std::vector<std::vector<void*>> grad_bufs(num_stages);
  
  // Pipeline warmup: run forward passes
  for (int b = 0; b < num_stages; ++b) {
    int mb_idx, stage;
    PipelineStage pstage;
    if (ctx.next_step(&mb_idx, &pstage) && pstage == PipelineStage::Forward) {
      forward_fn(mb_idx, streams[b]);
    }
  }
  
  // Steady state: 1F1B
  bool done = false;
  while (!done) {
    done = true;
    
    for (int stage = 0; stage < num_stages; ++stage) {
      int mb_idx, pstage_val;
      PipelineStage pstage;
      
      if (ctx.next_step(&mb_idx, &pstage)) {
        done = false;
        
        if (pstage == PipelineStage::Forward) {
          forward_fn(mb_idx, streams[stage]);
        } else {
          backward_fn(mb_idx, streams[stage]);
        }
      }
    }
  }
  
  // Cleanup
  for (auto& s : streams) {
    cudaStreamDestroy(s);
  }
}

}  // namespace cuda::distributed
```

### Integration with NCCL Communicator Split

For multi-stage pipeline parallelism, split NCCL communicator:

```cpp
// Create pipeline-specific communicator group
void init_pipeline_communicators(int num_stages) {
  // Split by pipeline stage
  for (int stage = 0; stage < num_stages; ++stage) {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    
    NCCL_CHECK(ncclCommSplit(
      global_comm_,           // Parent communicator
      stage,                  // Color = stage ID
      stage,                  // Key = stage ID (preserve order)
      &pipeline_comms_[stage],
      &config
    ));
  }
}
```

---

## 4. Version Compatibility Matrix

### NCCL + CUDA Version Requirements

| NCCL | CUDA 12.x | CUDA 17+ | CUDA 20 | Notes |
|------|-----------|----------|---------|-------|
| **2.25** | Full | Full | Full | Recommended baseline |
| **2.26** | Full | Full | Full | Bug fixes |
| **2.27** | Full | Full | Full | Performance improvements |
| **2.28** | Full | Full | Full | SHARP enhancements |
| **2.29** | Full | Full | Full | New communicator APIs |
| **2.30** | Full | Full | Full | Current latest, full support |

### C++ Standard Compatibility

| C++ Version | NCCL Headers | CUDA 20 | Nova Code |
|-------------|--------------|---------|-----------|
| C++17 | Supported | Supported | Not used |
| **C++20** | Supported | Supported | `std::jthread`, concepts |
| **C++23** | Supported | Supported | `std::expected`, constexpr |

**NCCL Header C++ Compatibility:**
```cpp
// NCCL headers are C, but work with C++ compilers
extern "C" {
#include <nccl.h>
}

// Nova C++ wrappers use modern C++ features
namespace cuda::collective {
  using std::expected;  // C++23, use std::optional for C++20
}
```

### NVLink / PCIe Detection

```cpp
// Utility to detect available bandwidth
enum class LinkType {
  NVLink,   // ~300 GB/s per link
  PCIe,     // ~32 GB/s Gen4 x16
  CPU,      // ~10 GB/s, host-mediated
  None      // Not accessible
};

LinkType detect_link_type(int device_a, int device_b) {
  // Check NVLink connections
  int can_access;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_a, device_b));
  
  if (!can_access) return LinkType::None;
  
  // For detailed topology, use nvidia-smi or NVML
  // nvmlDeviceGetCurrPcieLinkWidth() / nvmlDeviceGetPcieMaxLinkWidth()
  
  // Assume NVLink if peer access works (simplified)
  // Real implementation: parse nvidia-smi topo output
  return LinkType::NVLink;  // Conservative default
}
```

---

## 5. Integration with Existing Architecture

### Updated Directory Structure

```
include/cuda/
  collective/                   # NEW: NCCL integration
  │   ├── nccl_context.h       # NCCL communicator management
  │   ├── nccl_ops.h           # Collective operation wrappers
  │   └── types.h              # NCCL type mappings
  distributed/
  │   ├── tensor_parallel.h    # ENHANCED: Full TP implementation
  │   ├── pipeline.h           # NEW: Pipeline parallelism
  │   └── common.h             # ENHANCED: Shared utilities

src/cuda/
  collective/
  │   ├── nccl_context.cpp     # NCCL init/fini
  │   └── nccl_ops.cu          # NCCL kernels
  distributed/
  │   ├── tensor_ops.cu        # Split strategies
  │   └── pipeline.cu          # Pipeline scheduling
```

### CMake Integration Points

```cmake
# Extend existing cuda_multigpu with new components
target_sources(cuda_multigpu PRIVATE
  ${CMAKE_SOURCE_DIR}/src/cuda/collective/nccl_context.cpp
  ${CMAKE_SOURCE_DIR}/src/cuda/collective/nccl_ops.cu
  ${CMAKE_SOURCE_DIR}/src/cuda/distributed/tensor_ops.cu
  ${CMAKE_SOURCE_DIR}/src/cuda/distributed/pipeline.cu
)

if(NCCL_FOUND)
  target_link_libraries(cuda_multigpu PUBLIC NCCL::nccl)
endif()
```

### API Extensions

```cpp
// Extended DistributedMatmulOptions
struct DistributedMatmulOptions {
  enum class Strategy {
    DataParallel,       // Row-partition input (existing)
    TensorParallelCol,  // Column-partition weight (NEW)
    TensorParallelRow,  // Row-partition input (NEW)
    PipelineParallel    // Pipeline stages (NEW)
  };
  
  Strategy strategy = Strategy::DataParallel;
  int num_partitions = 1;
  
  // Pipeline-specific
  int num_micro_batches = 1;
  int micro_batch_size = 4;
  
  // Communication
  bool use_nccl = true;  // vs P2P fallback
  bool profile_comm = false;
};
```

---

## 6. What NOT to Add

| Technology | Why Avoid | When to Revisit |
|------------|-----------|-----------------|
| **Megatron-LM** | Over-engineered for single-node, PyTorch-centric API, large dependency | Multi-node training requirement |
| **DeepSpeed ZeRO** | Designed for multi-node, complex integration, state management overhead | Multi-node with memory constraints |
| **Horovod** | MPI-based, designed for TensorFlow/PyTorch, unnecessary for single-node | Multi-node, multiple frameworks |
| **NCCL persistent collectives** | Requires InfiniBand/nvlink fabric, complex setup | Production multi-node cluster |
| **GDRCopy** | GPU-to-network RDMA, specific hardware | InfiniBand deployment |

---

## 7. Testing Strategy

### NCCL-Specific Tests

```cpp
// test/collective/nccl_tests.cpp

TEST(NcclContext, InitializeAllDevices) {
  auto& ctx = NcclContext::instance();
  ctx.initialize({0, 1});  // Or all available devices
  
  EXPECT_EQ(ctx.rank_count(), 2);
  EXPECT_NO_THROW(ctx.comm(0));
  EXPECT_NO_THROW(ctx.comm(1));
}

TEST(NcclCollectives, AllReduce) {
  auto& ctx = NcclContext::instance();
  NcclCollectives coll(ctx.comm(0));
  
  std::vector<float> send(1024, 1.0f);
  std::vector<float> recv(1024, 0.0f);
  
  cuda::memory::unique_ptr<float> d_send = 
    cuda::memory::make_device_unique<float>(1024);
  cuda::memory::unique_ptr<float> d_recv = 
    cuda::memory::make_device_unique<float>(1024);
  
  CUDA_CHECK(cudaMemcpy(d_send.get(), send.data(), 
                        1024 * sizeof(float), cudaMemcpyHostToDevice));
  
  // This requires multi-GPU setup
  if (ctx.rank_count() > 1) {
    coll.all_reduce(d_send.get(), d_recv.get(), 1024,
                    ncclFloat32, ncclSum, 0);
    
    CUDA_CHECK(cudaMemcpy(recv.data(), d_recv.get(),
                          1024 * sizeof(float), cudaMemcpyDeviceToHost));
    
    EXPECT_EQ(recv[0], ctx.rank_count());  // Sum of 1s * num_ranks
  }
}

TEST(TensorParallel, ColumnSplit) {
  // Verify column-parallel produces same result as single-GPU
  constexpr int M = 128, K = 256, N = 512;
  
  auto single_gpu = reference_matmul(M, K, N);  // Single GPU result
  auto multi_gpu = column_parallel_matmul(M, K, N, 2);  // 2-GPU split
  
  ASSERT_NEAR(single_gpu.norm(), multi_gpu.norm(), 1e-3f);
}
```

---

## Sources

### NCCL Documentation (HIGH confidence)
- [NCCL 2.30 API Reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)
- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html)
- [NCCL Types and Data Types](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html)
- [CUDA Stream Semantics with NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)

### Tensor Parallelism (MEDIUM confidence)
- Megatron-LM: Column-parallel and row-parallel linear layers
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) — TE has TP patterns
- PyTorch DDP/FSDP patterns (for reference, not for integration)

### Pipeline Parallelism (MEDIUM confidence)
- [GPipe: Efficient Training of Giant Neural Networks via Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/abs/1806.03377)
- DeepSpeed pipeline parallelism implementation

---

*Stack research for: Nova CUDA Library v1.2+ NCCL and Parallelism Support*
*Researched: 2026-04-24*
