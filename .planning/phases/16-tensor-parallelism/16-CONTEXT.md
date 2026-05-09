# Phase 16: Tensor Parallelism - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>

## Phase Boundary

Implement tensor parallelism patterns for transformer layers with column-parallel and row-parallel matmul strategies. Each GPU holds a shard of the weight matrix rather than replicating it. Includes integration with existing DistributedMatmul and memory profiling for TP degree selection.

</domain>

<decisions>

## Implementation Decisions

### API Design

- Use TensorParallelMatmul class with strategy enum (ColumnParallel, RowParallel)
- Provide static factory methods for common patterns
- Follow existing DistributedMatmul naming conventions for consistency

### Column-Parallel Strategy

- Weight matrix B split along output dimension (n)
- Each GPU computes partial result: C_part = A @ B_part
- Use AllReduce to sum partial results across GPUs (not AllGather like DataParallel)
- Input A replicated on all GPUs
- Output C identical on all GPUs after reduction

### Row-Parallel Strategy

- Weight matrix B split along input dimension (k) for forward pass
- In practice, often implemented as column-parallel of transpose
- Each GPU computes C_part = A_part @ B_part
- No reduction needed (output rows are independent)
- Input A must be partitioned (requires preceding column-parallel layer)

### Layer Patterns

- ColumnParallelLayer: QKV projection (splits output dimension)
- RowParallelLayer: Output projection after column-parallel (splits input dimension)
- Both use all-reduce synchronization after matmul

### TP Degree Selection

- Auto-select based on available GPU memory
- Profile memory usage per TP degree
- Provide helper to estimate maximum TP degree

### Integration

- Extend DistributedMatmulOptions with TensorParallelStrategy
- Memory profiler reports working set per degree
- Works with existing NcclContext and DeviceMesh

### the agent's Discretion

All buffer allocation strategies, synchronization ordering, and error handling follow existing NCCL patterns.

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets

- DistributedMatmul for multi-GPU matmul patterns
- NcclAllReduce for result aggregation
- NcclCollective base class
- DeviceMesh for GPU enumeration
- MemoryPool for buffer management

### Established Patterns

- Static factory methods and synchronous wrappers
- Stream-based async variants with explicit cudaStream_t
- Non-copyable, movable class design
- Header includes for type definitions

### Integration Points

- New headers in include/cuda/neural/ (parallel to matmul.h)
- Source files in src/cuda/neural/
- Extend DistributedMatmulOptions enum
- Tests go in tests/cuda/neural/

</code_context>

<specifics>

## Specific Ideas

Column-parallel: C[:, n_start:n_end] = A @ B[:, n_start:n_end] per GPU, then AllReduce
Row-parallel: C[m_start:m_end, :] = A[m_start:m_end, :] @ B[:, :] per GPU (no sync)
Memory profiling: Track activation size, weight shard size, gradient buffer

</specifics>

<deferred>

## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
