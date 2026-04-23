# Technology Stack

**Mapped:** 2026-04-23

## Languages

| Language | Standard | Notes |
|----------|----------|-------|
| **C++** | C++20 | Primary language, requires C++20 features |
| **CUDA** | CUDA 17 | Device kernels, GPU programming |

## Build System

- **CMake** 3.25+ (minimum required)
- **Makefile** wrapper for common targets

## Dependencies

### CUDA Libraries

| Library | Usage | Header |
|---------|-------|--------|
| **CUDA Runtime** | Memory allocation, kernel launch | `<cuda_runtime.h>` |
| **cuBLAS** | Matrix operations | `<cublas_v2.h>` |

### External (FetchContent)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Google Test** | v1.14.0 | Unit testing framework |

### Internal Libraries

- `cuda_memory` - Interface library for memory layer
- `cuda_device` - Interface library for device kernels
- `cuda_algo` - Interface library for algorithm wrappers
- `cuda_api` - Interface library for high-level API
- `cuda_impl` - Static library containing all implementations

## CUDA Targets

The project targets CUDA architectures:
- 6.0 (Pascal)
- 7.0 (Volta)
- 8.0 (Turing)
- 9.0 (Ampere)

## Key Tools

| Tool | Purpose |
|------|---------|
| **clang-format** | Code formatting |
| **clang-tidy** | Static analysis |
| **pre-commit** | Git hooks (commit-msg, pre-commit, pre-push) |
| **Docker** | Containerized builds |

## Configuration

- **Compiler**: CMake CUDA/C++ compiler detection
- **Build type**: Release (default)
- **Separable compilation**: Enabled for device libraries
