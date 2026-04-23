# Architecture

**Mapped:** 2026-04-23

## Design Pattern

**Layered Architecture** with five distinct layers, each building upon the previous.

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: High-Level API (STL-style)                        │
│  cuda::reduce(), cuda::sort()                              │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Algorithm Wrappers                                │
│  cuda::algo::reduce_sum(), memory management               │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Device Kernels                                   │
│  Pure __global__ kernels, no memory allocation             │
└─────────────────────────────────────────────────────────────┘
                               ▲
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Memory Foundation                                │
│  Buffer<T>, unique_ptr<T>, MemoryPool, Allocator concepts   │
└─────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

| Layer | Namespace | Purpose | Dependencies |
|-------|-----------|---------|--------------|
| **Layer 0** | `cuda::memory` | Memory allocation, RAII, pooling | CUDA runtime |
| **Layer 1** | `cuda::device` | Pure device kernels | Layer 0 |
| **Layer 2** | `cuda::algo` | Memory management, algorithms | Layers 0, 1 |
| **Layer 3** | `cuda::api` | STL-style containers | Layers 0, 1, 2 |

## Key Abstractions

### Memory Management

- `cuda::memory::Buffer<T>` - RAII wrapper for device memory
- `cuda::memory::unique_ptr<T>` - Smart pointer for device memory
- `cuda::memory::MemoryPool` - Memory pool for efficient allocation

### Algorithm Interface

- Wrappers handle memory allocation/deallocation automatically
- Device pointers passed explicitly between layers
- Error handling via `CUDA_CHECK` macro

### Data Flow

```
Host Code
    │
    ▼
cuda::api::DeviceVector ──────► cuda::algo::reduce_sum()
    │                                    │
    │ (copy_from/to)                     ▼
    │                          cuda::device::reduce_kernel
    │                                    │
    ▼                                    ▼
Host Buffer                       Device Memory (Buffer)
```

## Error Handling

Custom exception hierarchy in `cuda::device` namespace:

- `CudaException` - CUDA runtime errors
- `CublasException` - cuBLAS errors
- `CUDA_CHECK()` macro - Automatic error checking
- `CUBLAS_CHECK()` macro - cuBLAS error checking

## Additional Modules

| Module | Namespace | Purpose |
|--------|-----------|---------|
| **Image** | (global) | Image processing (blur, sobel, morphology) |
| **Parallel** | (global) | Parallel primitives (scan, sort, histogram) |
| **Matrix** | (global) | Matrix operations (add, mult, ops) |
| **Convolution** | (global) | 2D convolution |
