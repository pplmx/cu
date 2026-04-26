# Architecture Overview

## Five-Layer Architecture

Nova follows a five-layer CUDA architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│              API Layer (api/)               │
│         High-level user interfaces          │
├─────────────────────────────────────────────┤
│            Algorithm Layer (algo/)          │
│       Parallel algorithms, reduce, scan      │
├─────────────────────────────────────────────┤
│             Device Layer (device/)          │
│       Device management, memory ops         │
├─────────────────────────────────────────────┤
│            Memory Layer (memory/)           │
│      Buffer management, memory pools        │
├─────────────────────────────────────────────┤
│         CUDA Runtime Integration           │
│        cuBLAS, cuFFT, cuSPARSE            │
└─────────────────────────────────────────────┘
```

## New in v2.2

### Performance Layer

- **Kernel Fusion** (`cuda/neural/fusion/`): Fused kernels for matmul + bias + activation
- **Autotuning** (`cuda/performance/autotuner.h`): Hardware-aware parameter optimization
- **Memory Optimization** (`cuda/memory_opt/`): Adaptive pool sizing and compression

### Neural Networks Layer

- **Transformer Components** (`cuda/neural/transformer/`):
  - Multi-head attention with configurable heads
  - Sinusoidal and learned positional encoding

- **Loss Functions** (`cuda/neural/loss/`):
  - Cross-entropy with numerical stability
  - Focal loss for class imbalance
  - Contrastive loss for representation learning

- **Optimizers** (`cuda/neural/optimizers/`):
  - AdamW with weight decay
  - LAMB with layer-wise adaptation
  - Gradient clipping utilities

### Tooling Layer

- **Bank Conflict Analyzer** (`cuda/tools/bank_conflict_analyzer.h`): Shared memory optimization
- **Timeline Visualizer** (`cuda/tools/timeline_visualizer.h`): Chrome trace export
- **Bandwidth Analyzer** (`cuda/tools/timeline_visualizer.h`): Memory bandwidth profiling

## Data Flow

```
User Input
    ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Memory     │ →  │  Device     │ →  │  Algorithm  │
│  Layer      │    │  Layer      │    │  Layer      │
└─────────────┘    └─────────────┘    └─────────────┘
                                              ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Output    │ ←  │   Loss      │ ←  │    Forward  │
│             │    │  Functions  │    │    Pass     │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Module Dependencies

```
api/
├── algo/          ← algorithm implementations
├── device/        ← device queries
├── memory/        ← buffer management
└── neural/        ← ML primitives
    ├── activation.h
    ├── matmul.h
    ├── softmax.h
    ├── transformer/
    │   └── attention.h    ← NEW
    ├── loss/
    │   └── loss_functions.h  ← NEW
    └── optimizers/
        └── optimizers.h   ← NEW

performance/
├── profiler.h
├── autotuner.h    ← NEW
└── device_info.h

memory_opt/
└── memory_optimizer.h  ← ENHANCED

tools/
├── bank_conflict_analyzer.h  ← NEW
└── timeline_visualizer.h    ← NEW
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Header-only utilities | Zero linking overhead, easy integration |
| Singleton managers | Centralized state, easy configuration |
| Stream-based async | Native CUDA concurrency model |
| cuBLAS for matmul | Optimized kernels, hardware acceleration |
| ZSTD compression | Fast, good compression ratio |
| Chrome trace format | Standard tooling support |
