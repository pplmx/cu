# Phase 21 Context: Checkpoint/Restart

## Phase Overview

**Goal:** Implement full state checkpoint/restart for training recovery.

## Requirements

| ID | Requirement | Success Criteria |
|----|-------------|------------------|
| CKPT-01 | CheckpointManager with async writes | Async writes don't block training |
| CKPT-02 | Full state serialization | Weights + optimizer states + RNG |
| CKPT-03 | Storage backend abstraction | Filesystem + future object store |
| CKPT-04 | Incremental checkpoint support | Only save changed tensors |
| CKPT-05 | Automatic checkpoint on error | Save before recovery attempts |

## Implementation Strategy

### Architecture

```
CheckpointManager (singleton)
├── StorageBackend (abstract)
│   ├── FileStorageBackend
│   └── ObjectStorageBackend (future)
├── StateSerializer
│   ├── ModelState
│   ├── OptimizerState
│   └── RNGState
└── CheckpointCoordinator
```

### Async Write Strategy

- Dedicated CUDA stream for checkpoint I/O
- Double-buffering to avoid blocking training
- cudaHostRegister for pinned memory transfers
- Async CUDA memcpy to pinned, then CPU write

### Full State Components

1. **ModelState**: All parameter tensors with shapes/dtypes
2. **OptimizerState**: Momentum, variance, etc. (Adam/SGD)
3. **RNGState**: CUDA RNG state for reproducibility

### Incremental Checkpoint Design

- Checksum/hash of tensor contents
- Only write tensors that changed since last checkpoint
- Maintain manifest of current checkpoint state
- Support for full checkpoint every N steps

### File Format

- Directory per checkpoint: `ckpt_step_1000/`
- `manifest.json`: Metadata (step, timestamp, version)
- `model.safetensors`: Model weights (safetensors format)
- `optimizer.safetensors`: Optimizer states
- `rng.pt`: RNG state
- `incremental/`: Changed tensors since last checkpoint

## Dependencies

- Existing `cuda/memory/Buffer.h` for tensor storage
- Existing `cuda/memory/MemoryPool.h` for allocations
- CUDA 20+ async APIs

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Blocking I/O | Dedicated async stream |
| Large checkpoint size | Compression, incremental |
| Partial writes | Atomic rename |
| Cross-rank sync | Coordinated checkpoint |

## Gray Areas

1. **Compression**: LZ4 vs ZSTD trade-off (speed vs size)
2. **Checkpoint frequency**: Configurable interval vs auto-trigger
3. **Storage quota**: How many checkpoints to keep
