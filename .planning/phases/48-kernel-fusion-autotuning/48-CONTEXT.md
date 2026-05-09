---
phase_number: 48
phase_name: Kernel Fusion & Autotuning
status: in_progress
created: 2026-04-27
requirements:
  - PERF-01
  - PERF-03
---

# Phase 48: Kernel Fusion & Autotuning - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning
**Mode:** Auto-generated (discuss skipped for v2.2)

## Phase Boundary

Enable kernel fusion for chained operations and autotuning infrastructure.

## Success Criteria

1. User can fuse matmul + bias + activation into single kernel
2. User can configure fusion patterns via policy
3. User can run autotuning for block sizes on target GPU
4. Autotuned parameters persist in config file

## Key Decisions

- Use CUDA fusion API where available
- Provide fallback manual fusion for older CUDA versions
- Autotuning uses grid search with warmup runs

## Existing Code Insights

- `include/cuda/neural/fusion/kernel_fusion.h` - Existing basic fusion primitives
- `include/cuda/performance/device_info.h` - Device capability queries
- `include/cuda/neural/matmul.h` - Existing matmul operations

## Implementation Approach

### PERF-01: Fused matmul + bias + activation

Enhance existing `kernel_fusion.h` with:

- `FusedMatmulBiasAct` class supporting configurable activation
- CUDA fusion API integration with cublasLt for modern GPUs
- Manual fallback fusion for older CUDA versions

### PERF-03: Autotuning infrastructure

Create new `include/cuda/performance/autotuner.h`:

- `Autotuner` class with device-specific parameter discovery
- Grid search over block sizes with warmup runs
- Config persistence to `autotune_config.json`
- Integration with existing `DeviceProperties`

## Dependencies

- None (foundation phase)
- Works with existing matmul and device info infrastructure
