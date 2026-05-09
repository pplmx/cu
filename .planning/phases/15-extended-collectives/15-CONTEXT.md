# Phase 15: Extended Collectives - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>

## Phase Boundary

Implement advanced NCCL collective operations (AllGather, ReduceScatter), group batching with ncclGroupStart/End, and a unified fallback path for deployment flexibility. Communicator caching reduces initialization overhead for repeated operations.

This phase extends Phase 14's Core Collectives with additional primitives used in distributed training workloads.

</domain>

<decisions>

## Implementation Decisions

### API Consistency

- Use same NcclCollective base class pattern as NcclAllReduce, NcclBroadcast, NcclBarrier (Phase 14)
- Provide both async (with stream) and sync (blocking) variants
- Mirror the NcclResult return type pattern for error handling

### AllGather Behavior

- Fixed-size all-gather with explicit output buffer sized for all ranks
- Output buffer layout: contiguous per-rank segments (same as NCCL default)

### ReduceScatter Behavior

- Support configurable root for alternative aggregation patterns
- Use ncclReduceScatter with contiguous input/output buffer model

### Group Operations

- NcclGroupHandle class for batching multiple collectives
- RAII pattern with constructor/destructor for automatic begin/end
- Thread-safe dispatch to prevent cross-collective deadlocks

### Unified Fallback

- Detect NCCL availability via NcclContext::has_nccl()
- When NCCL unavailable, fall back to P2P implementations from v1.1
- Unified API: same signatures whether NCCL or P2P path taken

### Communicator Caching

- Cache communicators by device count and rank configuration
- Use NCCL's built-in communicator caching where available (NCCL 2.26+)
- LRU eviction when cache limit reached (configurable, default 16 entries)

### the agent's Discretion

All other implementation choices (buffer alignment, error recovery, memory allocation) at agent's discretion per codebase patterns.

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets

- NcclCollective base class (nccl_collective.h) - provides context access, communicator lookup, stream management
- NcclResult return type for error handling
- safe_nccl_call wrapper for async error detection
- Existing P2P implementations in v1.1 (distributed/multi_gpu_ops.h)

### Established Patterns

- Stream-ordered async operations with cudaStream_t parameter
- Both _async (non-blocking) and sync (blocking) variants
- Non-copyable, movable class design
- Header-only API with implementation in .cpp files

### Integration Points

- New headers go in include/cuda/nccl/
- Source files go in src/cuda/nccl/
- CMake: link against cuda_nccl library (Phase 13)
- Tests go in tests/cuda/nccl/test_nccl_extended.cpp

</code_context>

<specifics>

## Specific Ideas

AllGather: Output buffer must be (device_count * send_count) elements
ReduceScatter: Input buffer must be (device_count * recv_count) elements
Group ops: Must handle mixed collective types in single group
Fallback: Preserve P2P semantics for backwards compatibility

</specifics>

<deferred>

## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
