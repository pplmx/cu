# Phase 17: Pipeline Parallelism - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement pipeline parallelism scheduling for deep model training with 1F1B schedule, P2P communication primitives, and activation buffer management. Includes communicator splitting for hybrid TP+DP parallelism and interleaved scheduling to reduce bubble overhead.

</domain>

<decisions>
## Implementation Decisions

### 1F1B Schedule
- Classic pipe schedule: 1 forward microbatch, 1 backward microbatch
- Overlaps forward and backward passes across stages
- Bubble overhead proportional to pipeline depth (K) and batch size (M)
- M >= 4K recommended to hide bubble overhead

### P2P Primitives
- Use cudaMemcpyAsync with peer access for inter-stage communication
- Send activations forward (layer n -> layer n+1)
- Send gradients backward (layer n -> layer n-1)
- Double-buffering with ping-pong buffers to hide communication

### Activation Buffer Management
- Two buffers per stage: ping and pong
- While GPU computes on ping, receive into pong
- Swap roles each microbatch
- Reduces idle time during communication

### Communicator Splitting
- Use ncclCommSplit for hybrid parallelism
- One communicator per TP group
- One communicator per DP rank
- Avoids deadlocks from mixed collective operations

### Interleaved Schedule
- Alternative to standard 1F1B
- Multiple micro batches per stage before backward
- Reduces bubble overhead but increases memory
- Configurable via schedule type enum

### Stage Balance Validation
- Profile compute time per stage
- Report variance across stages
- Warn if variance > 10%
- Suggest rebalancing or stage merging

### the agent's Discretion
All implementation details (buffer sizes, synchronization patterns, error handling) follow existing codebase patterns.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- NcclContext and DeviceMesh for multi-GPU coordination
- NcclAllReduce, NcclBroadcast for collectives
- MemoryPool for buffer management
- TensorParallelMatmul for TP integration

### Established Patterns
- RAII classes with constructor/destructor lifecycle
- Async/sync method pairs
- Static factory methods where appropriate

### Integration Points
- Headers in include/cuda/pipeline/
- Source files in src/cuda/pipeline/
- Works with TensorParallelLayer from Phase 16

</code_context>

<specifics>
## Specific Ideas

1F1B: forward(mb), backward(mb) repeated per stage
P2P: cudaMemcpyAsync with peer GPU directly
Ping-pong: Buffer A and Buffer B, swap on each microbatch
Comm split: ncclCommSplit(comm, color, key) creates sub-comms

</specifics>

<deferred>
## Deferred Ideas

None — all Phase 17 requirements addressed.

</deferred>
