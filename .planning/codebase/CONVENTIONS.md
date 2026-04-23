# Coding Conventions

**Mapped:** 2026-04-23

## Style Guide

**Based on:** Google C++ Style Guide with project-specific overrides

### Formatting

| Setting | Value |
|---------|-------|
| **Standard** | C++20 |
| **Column limit** | 180 |
| **Indent** | 4 spaces (no tabs) |
| **Braces** | Attached |
| **Pointer alignment** | Left |

### Code Style Rules

- **Short if statements**: Never on single line
- **Short functions**: Allowed inline
- **Short loops**: Never on single line
- **Constructor initializers**: Break before colon
- **Namespace indentation**: All (indent inside namespaces)

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| **Namespaces** | lowercase, nested | `cuda::memory` |
| **Classes** | PascalCase | `Buffer<T>`, `CudaException` |
| **Enums** | PascalCase | `ReduceOp` |
| **Enum values** | PascalCase | `ReduceOp::SUM` |
| **Methods** | PascalCase | `copy_from()`, `release()` |
| **Members** | snake_case with underscore | `data_`, `size_` |
| **Constants** | PascalCase | `WARP_SIZE` |
| **Macros** | SCREAMING_SNAKE | `CUDA_CHECK()` |

## Header Organization

```cpp
#pragma once

#include <standard_libs>

#include "local/lib.h"

namespace cuda::memory {
    // Declarations...
}  // namespace cuda::memory
```

### Include Order

1. `#pragma once` (header files only)
2. Standard library includes
3. Third-party includes
4. Local includes
5. Namespace declarations

## Documentation

All public APIs should have Doxygen-style documentation:

```cpp
/**
 * @class Buffer
 * @brief RAII wrapper for CUDA device memory with automatic memory management.
 *
 * Buffer<T> provides RAII semantics for GPU memory allocation and deallocation.
 * Memory is automatically freed when the Buffer goes out of scope.
 *
 * @tparam T The element type stored in the buffer
 *
 * @example
 * @code
 * cuda::memory::Buffer<int> buf(1024);
 * @endcode
 */
```

## Error Handling

### Macros

```cpp
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            throw cuda::device::CudaException(err, __FILE__, __LINE__); \
        }                                                               \
    } while (0)
```

### Custom Exceptions

```cpp
class CudaException : public std::runtime_error {
public:
    explicit CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(format_error(err, file, line)),
          error_(err) {}
};
```

## RAII Patterns

All resource-owning types follow RAII:

```cpp
class Buffer {
public:
    ~Buffer() {
        if (data_) {
            cudaFree(data_);
        }
    }
    
    // Delete copy operations
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    // Allow move operations
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
};
```

## Kernel Patterns

Device kernels follow NVIDIA conventions:

```cpp
template <typename T>
__device__ T warp_reduce(T val, ReduceOp op) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // ...
    }
    return val;
}
```

## Pre-commit Hooks

The project uses `pre-commit` with hooks for:
- `commit-msg` - Validate commit messages
- `pre-commit` - Run linting/formatting
- `pre-push` - Run tests before push
