# Phase 59: CUDA Graphs Foundation - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning
**Mode:** Autonomous execution

## Phase Boundary

Implement CUDA Graphs foundation for batch workload optimization. CUDA Graphs reduce kernel launch overhead by 10-50x by capturing compute graphs and replaying them.

## Requirements

- GRAPH-01: GraphExecutor Core - capture, update, launch with 64 parameter support
- GRAPH-02: Memory Node Integration - device, host-pinned, managed memory nodes
- GRAPH-03: Algorithm Graph Wrappers - wrap reduce, scan, sort for graph execution

## Implementation Decisions

### Architecture

- Create `cuda/production/` namespace for production hardening features
- `GraphExecutor` class as the main interface
- RAII pattern for graph lifecycle management
- Integration with existing `cuda::stream::Stream` class

### CUDA Graphs API Usage

- Use `cudaGraphCreate()`, `cudaGraphInstantiate()`
- Capture via `cudaStreamBeginCapture()`, `cudaStreamEndCapture()`
- Replay via `cudaGraphLaunch()`
- Parameter nodes via `cudaGraphAddNode()` with `cudaKernelNodeParams`

### Memory Integration

- Device memory nodes: `cudaGraphAddMemcpyNode()`
- Host-pinned memory: `cudaHostRegister()` + memcpy nodes
- Managed memory: `cudaMallocManaged()` + implicit capture

## Codebase Alignment

- Follow existing patterns in `include/cuda/stream/stream.h`
- Use `CUDA_CHECK` macro from `include/cuda/device/error.h`
- RAII destructors with proper cleanup
- Move semantics for resource management

## Specific Ideas

1. **GraphExecutor class** with:
   - Constructor taking optional stream
   - `begin_capture()` → starts stream capture
   - `end_capture()` → finishes and validates graph
   - `instantiate()` → creates executable graph
   - `launch()` → replays graph
   - `update_param()` → updates kernel parameters without rebuild

2. **MemoryNode helper** for capturing allocations

3. **AlgoWrapper template** for wrapping algorithms
