# Directory Structure

**Mapped:** 2026-04-23

## Top-Level Structure

```
nova/
├── include/              # Public headers (layer interface)
│   ├── cuda/            # CUDA library (4-layer architecture)
│   │   ├── memory/      # Layer 0: Buffer, unique_ptr, MemoryPool
│   │   ├── device/      # Layer 1: Kernels, device utils
│   │   ├── algo/        # Layer 2: Algorithm wrappers
│   │   └── api/         # Layer 3: High-level containers
│   ├── image/           # Image processing module
│   ├── parallel/        # Parallel primitives
│   ├── matrix/          # Matrix operations
│   └── convolution/     # Convolution
├── src/                 # Implementation files
│   ├── cuda/           # CUDA implementations
│   │   ├── device/     # Device kernel implementations
│   │   └── algo/       # Algorithm implementations
│   ├── image/          # Image processing implementations
│   ├── parallel/       # Parallel primitive implementations
│   ├── matrix/         # Matrix implementations
│   ├── convolution/    # Convolution implementations
│   ├── memory/         # Memory pool implementations
│   └── main.cpp        # Demo application
├── tests/              # Test files
│   ├── *_test.cu       # CUDA tests (Google Test)
│   └── *_test.cpp      # C++ tests (Google Test)
├── cmake/              # CMake modules
├── .github/            # GitHub configuration
│   └── workflows/      # CI/CD pipelines
├── data/               # Demo data files
├── build/              # Build output (gitignored)
└── docs/               # Documentation
```

## Key File Locations

### Memory Layer (Layer 0)

| File | Purpose |
|------|---------|
| `include/cuda/memory/buffer.h` | Buffer<T> RAII wrapper |
| `include/cuda/memory/unique_ptr.h` | Smart pointer for device memory |
| `include/cuda/memory/memory_pool.h` | Memory pool interface |
| `src/memory/buffer.cpp` | Buffer implementation |
| `src/memory/memory_pool.cpp` | MemoryPool implementation |

### Device Layer (Layer 1)

| File | Purpose |
|------|---------|
| `include/cuda/device/device_utils.h` | warp_reduce, block_reduce |
| `include/cuda/device/error.h` | CUDA_CHECK, exceptions |
| `include/cuda/device/reduce_kernels.h` | Kernel declarations |
| `src/cuda/device/reduce_kernels.cu` | Kernel implementations |

### Algorithm Layer (Layer 2)

| File | Purpose |
|------|---------|
| `include/cuda/algo/reduce.h` | reduce_sum, reduce_max, reduce_min |
| `src/cuda/algo/reduce.cu` | Algorithm implementations |

### API Layer (Layer 3)

| File | Purpose |
|------|---------|
| `include/cuda/api/device_vector.h` | STL-style container |
| `include/cuda/api/stream.h` | Stream wrapper |
| `include/cuda/api/config.h` | Algorithm configuration |

## Naming Conventions

| Pattern | Example |
|---------|---------|
| Headers | `*.h`, `*.hpp` |
| CUDA sources | `*.cu`, `*.cuh` |
| C++ sources | `*.cpp` |
| Tests | `*_test.cu`, `*_test.cpp` |

## Module Headers

### CUDA Module

```cpp
#include "cuda/memory/buffer.h"
#include "cuda/device/device_utils.h"
#include "cuda/algo/reduce.h"
```

### Other Modules

```cpp
#include "image/brightness.h"
#include "parallel/scan.h"
#include "matrix/mult.h"
#include "convolution/conv2d.h"
```
