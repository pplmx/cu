# Phase 48: Kernel Fusion & Autotuning - Plan

## Requirements

- **PERF-01**: User can fuse matmul + bias + activation into single kernel
- **PERF-03**: User can run autotuning for block sizes on target GPU

## Implementation

### PERF-01: Fused Matmul + Bias + Activation

#### Files to Create

1. `include/cuda/neural/fusion/fused_matmul_bias_act.h`
   - `FusedMatmulBiasAct` class with configurable activation
   - `FusionPolicyManager` singleton for policy configuration
   - Activation type enum (None, ReLU, Sigmoid, Tanh, GELU)
   - Template activation kernel functions

2. `src/neural/fusion/fused_matmul_bias_act.cpp`
   - `FusedMatmulBiasAct::forward()` implementation
   - Fallback path using cuBLAS + element-wise kernels
   - `FusionPolicyManager` implementation
   - CUDA kernel implementations for activations

3. `tests/neural/fusion/fused_matmul_bias_act_test.cpp`
   - Tests for matmul + bias + ReLU
   - Tests for matmul + bias + Sigmoid
   - Tests for matmul + bias + GELU
   - Tests for FusionPolicyManager
   - Tests for activation type switching

### PERF-03: Autotuning Infrastructure

#### Files to Create

1. `include/cuda/performance/autotuner.h`
   - `Autotuner` class with configurable search space
   - `AutotuneConfig` with block_sizes, grid_sizes, iterations
   - `AutotuneResult` with optimal parameters
   - `AutotuneRegistry` singleton for persistent caching

2. `src/performance/autotuner.cpp`
   - `Autotuner::tune()` with grid search and warmup
   - Cache persistence to `autotune_config.json`
   - `AutotuneRegistry` implementation

3. `tests/performance/autotuner_test.cpp`
   - Basic autotuning test
   - Config setter tests
   - Cache operations test
   - Registry operations test

## Verification

1. **Build verification:**
   ```bash
   cmake --build build --parallel
   ```

2. **Test execution:**
   ```bash
   ctest -R "fusion|autotuner" -V
   ```

3. **Feature verification:**
   - `FusedMatmulBiasAct::forward()` produces correct output
   - `FusionPolicyManager` correctly enables/disables fusion
   - `Autotuner::tune()` returns valid results
   - Cache persists between sessions
