# CUDA Parallel Algorithms Library - Clean Architecture Design

**Date:** 2025-04-22
**Status:** Design
**Type:** Architecture (Greenfield Refactoring)

## 1. Design Principles

1. **Device-only API**: All public functions take `Buffer<T>` device pointers
2. **User manages memory**: Users create/manage Buffer lifecycle
3. **Single launch mechanism**: KernelLauncher is the only way to launch kernels
4. **No global state**: No `__constant__` globals, no static variables
5. **Encapsulated resources**: cuBLAS handles wrapped in RAII classes
6. **Fail-fast**: Exception-based error handling throughout

## 2. Final Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Public API (cuda::<Domain>::<Operation>)                              │
│                                                                         │
│  cuda::image::gaussianBlur(Buffer<uint8_t>& in, Buffer<uint8_t>& out)   │
│  cuda::image::brightness(Buffer<uint8_t>& img, float alpha, float beta)│
│  cuda::image::sobel(Buffer<uint8_t>& in, Buffer<uint8_t>& out)         │
│                                                                         │
│  cuda::matrix::multiply(const Buffer<T>& a, const Buffer<T>& b,          │
│                        Buffer<T>& c, int m, int n, int k)              │
│  cuda::matrix::add(const Buffer<T>& a, const Buffer<T>& b,             │
│                   Buffer<T>& c, int rows, int cols)                    │
│  cuda::matrix::transpose(const Buffer<T>& in, Buffer<T>& out)           │
│                                                                         │
│  cuda::parallel::exclusiveScan(const Buffer<T>& in, Buffer<T>& out)      │
│  cuda::parallel::sort(Buffer<T>& data)                                  │
│  cuda::parallel::histogram(const Buffer<uint8_t>& img, Buffer<int>& hist) │
│                                                                         │
│  cuda::convolution::conv2d(const Buffer<T>& in, const Buffer<T>& kernel, │
│                           Buffer<T>& out, const Conv2DConfig& config)  │
└───────────────────────────────────────────────────────────────────────┬───┘
                                                                      │
                              ▲                                        │
                              │                                        │
┌─────────────────────────────────────────────────────────────────────▼───┐
│  Kernel Launch Layer (cuda::algo)                                       │
│                                                                       │
│  class KernelLauncher {                                                │
│      grid(dim3).block(dim3).shared(size_t).stream(cudaStream_t)         │
│      launch(kernel, args...)                                           │
│      synchronize()                                                    │
│  }                                                                   │
│                                                                       │
│  class CublasHandle {  // RAII wrapper for cuBLAS                      │
│      cublasHandle_t get();                                             │
│  }                                                                    │
└───────────────────────────────────────────────────────────────────────┬───┘
                                                                      │
                              ▲                                        │
                              │                                        │
┌─────────────────────────────────────────────────────────────────────▼───┐
│  Memory Layer (cuda::memory)                                           │
│                                                                       │
│  Buffer<T> - owns device memory, RAII                                  │
│  unique_ptr<T> - single object ownership                              │
│  MemoryPool - pre-allocated block pool                                │
└───────────────────────────────────────────────────────────────────────┘
```

## 3. KernelLauncher

```cpp
// include/cuda/algo/kernel_launcher.h
#pragma once

#include <cuda_runtime.h>
#include "cuda/device/error.h"

namespace cuda::algo {

class KernelLauncher {
public:
    KernelLauncher() = default;

    KernelLauncher& grid(dim3 g) { grid_ = g; return *this; }
    KernelLauncher& grid(int x, int y = 1, int z = 1) { grid_ = dim3(x, y, z); return *this; }

    KernelLauncher& block(dim3 b) { block_ = b; return *this; }
    KernelLauncher& block(int x, int y = 1, int z = 1) { block_ = dim3(x, y, z); return *this; }

    KernelLauncher& shared(size_t bytes) { shared_ = bytes; return *this; }
    KernelLauncher& stream(cudaStream_t s) { stream_ = s; return *this; }

    template<typename Kernel, typename... Args>
    void launch(Kernel* kernel, Args&&... args) {
        kernel<<<grid_, block_, shared_, stream_>>>(std::forward<Args>(args)...);
        CUDA_CHECK(cudaGetLastError());
    }

    void synchronize() const {
        if (stream_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    dim3 grid() const { return grid_; }
    dim3 block() const { return block_; }
    size_t shared() const { return shared_; }
    cudaStream_t stream() const { return stream_; }

private:
    dim3 grid_{1, 1, 1};
    dim3 block_{1, 1, 1};
    size_t shared_{0};
    cudaStream_t stream_{nullptr};
};

// Helper: Calculate optimal grid for 2D compute
[[nodiscard]] constexpr dim3 calc_grid_2d(size_t width, size_t height,
                                         dim3 block = {16, 16, 1}) {
    return dim3((width + block.x - 1) / block.x,
               (height + block.y - 1) / block.y);
}

// Helper: Calculate optimal grid for 1D compute
[[nodiscard]] constexpr dim3 calc_grid_1d(size_t n, size_t block = 256) {
    return dim3((n + block - 1) / block);
}

}
```

## 4. CublasHandle (RAII wrapper)

```cpp
// include/cuda/algo/cublas_handle.h
#pragma once

#include <cublas_v2.h>
#include "cuda/device/error.h"

namespace cuda::algo {

class CublasHandle {
public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }

    ~CublasHandle() {
        cublasDestroy(handle_);  // Ignore error in destructor
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cublasDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cublasHandle_t get() const { return handle_; }
    [[nodiscard]] operator cublasHandle_t() const { return handle_; }

private:
    cublasHandle_t handle_{nullptr};
};

}
```

## 5. Image Module

```cpp
// include/cuda/image/filters.h
#pragma once

#include "cuda/memory/buffer.h"
#include "cuda/algo/kernel_launcher.h"

namespace cuda::image {

struct GaussianBlurConfig {
    float sigma = 1.0f;
    int kernel_size = 3;
};

void gaussian_blur(const memory::Buffer<uint8_t>& input,
                  memory::Buffer<uint8_t>& output,
                  size_t width, size_t height,
                  const GaussianBlurConfig& config = {});

struct BrightnessConfig {
    float alpha = 1.0f;  // contrast factor
    float beta = 0.0f;   // brightness offset
};

void brightness(const memory::Buffer<uint8_t>& input,
               memory::Buffer<uint8_t>& output,
               size_t width, size_t height,
               const BrightnessConfig& config = {});

void sobel_edge(const memory::Buffer<uint8_t>& input,
               memory::Buffer<uint8_t>& output,
               size_t width, size_t height,
               float threshold = 30.0f);

}
```

```cpp
// src/image/filters.cu
#include "image/filters.h"
#include <cmath>
#include <vector>
#include <stdexcept>

namespace cuda::image {
namespace {

__global__ void brightness_kernel(const uint8_t* input, uint8_t* output,
                                  size_t width, size_t height,
                                  float alpha, float beta) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const size_t idx = (y * width + x) * 3;

    for (int c = 0; c < 3; ++c) {
        float value = alpha * static_cast<float>(input[idx + c]) + beta;
        output[idx + c] = static_cast<uint8_t>(
            std::clamp(value, 0.0f, 255.0f));
    }
}

__global__ void sobel_x_kernel(const uint8_t* input, float* grad_x,
                               size_t width, size_t height) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        grad_x[y * width + x] = 0;
        return;
    }

    float gx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const uint8_t pixel = input[(y + dy) * width + dx];
            const int8_t sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
            gx += pixel * sobel_x[dy + 1][dx + 1];
        }
    }
    grad_x[y * width + x] = gx;
}

}

void brightness(const memory::Buffer<uint8_t>& input,
               memory::Buffer<uint8_t>& output,
               size_t width, size_t height,
               const BrightnessConfig& config) {

    dim3 block(16, 16);
    dim3 grid = calc_grid_2d(width, height, block);

    algo::KernelLauncher launcher;
    launcher.grid(grid).block(block);
    launcher.launch(brightness_kernel,
                   input.data(), output.data(),
                   width, height,
                   config.alpha, config.beta);
    launcher.synchronize();
}

void sobel_edge(const memory::Buffer<uint8_t>& input,
               memory::Buffer<uint8_t>& output,
               size_t width, size_t height,
               float threshold) {

    memory::Buffer<float> grad_x(width * height);
    memory::Buffer<float> grad_y(width * height);

    dim3 block(16, 16);
    dim3 grid = calc_grid_2d(width, height, block);

    algo::KernelLauncher launcher;
    launcher.grid(grid).block(block);

    // Compute gradients
    launcher.launch(sobel_x_kernel, input.data(), grad_x.data(), width, height);
    launcher.synchronize();

    // ... compute magnitude and threshold
}

}
```

## 6. Matrix Module

```cpp
// include/cuda/matrix/ops.h
#pragma once

#include "cuda/memory/buffer.h"
#include "cuda/algo/kernel_launcher.h"
#include "cuda/algo/cublas_handle.h"

namespace cuda::matrix {

void add(const memory::Buffer<float>& a,
         const memory::Buffer<float>& b,
         memory::Buffer<float>& c,
         int rows, int cols);

void multiply(const memory::Buffer<float>& a,
              const memory::Buffer<float>& b,
              memory::Buffer<float>& c,
              int m, int n, int k);

void transpose(const memory::Buffer<float>& input,
               memory::Buffer<float>& output,
               int rows, int cols);

void scale(memory::Buffer<float>& data, float scalar, int size);

}
```

```cpp
// src/matrix/ops.cu
#include "matrix/ops.h"

namespace cuda::matrix {
namespace {

__global__ void add_kernel(const float* a, const float* b, float* c,
                            int rows, int cols) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(rows * cols)) {
        c[idx] = a[idx] + b[idx];
    }
}

}

void add(const memory::Buffer<float>& a,
         const memory::Buffer<float>& b,
         memory::Buffer<float>& c,
         int rows, int cols) {

    algo::KernelLauncher launcher;
    launcher.grid(calc_grid_1d(rows * cols)).block(256);
    launcher.launch(add_kernel, a.data(), b.data(), c.data(), rows, cols);
    launcher.synchronize();
}

void multiply(const memory::Buffer<float>& a,
              const memory::Buffer<float>& b,
              memory::Buffer<float>& c,
              int m, int n, int k) {

    const float alpha = 1.0f, beta = 0.0f;
    algo::CublasHandle handle;

    // C = alpha * A * B + beta * C
    // Note: cuBLAS is column-major, we pass transposed dimensions
    CUBLAS_CHECK(cublasSgemm(handle.get(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        k, m, n,
        &alpha,
        b.data(), k,
        a.data(), m,
        &beta,
        c.data(), m));
}

}
```

## 7. Parallel Module

```cpp
// include/cuda/parallel/primitives.h
#pragma once

#include "cuda/memory/buffer.h"
#include <stdexcept>

namespace cuda::parallel {

constexpr size_t MAX_SCAN_SIZE = 1024;

class ScanSizeException : public std::invalid_argument {
public:
    explicit ScanSizeException(size_t size)
        : std::invalid_argument("Scan size " + std::to_string(size) +
                                " exceeds maximum " + std::to_string(MAX_SCAN_SIZE)) {}
};

void exclusive_scan(const memory::Buffer<int>& input,
                   memory::Buffer<int>& output,
                   size_t size);

template<typename T>
void sort(memory::Buffer<T>& data, size_t size);

}
```

## 8. Files Structure

```
include/
├── cuda/
│   ├── memory/
│   │   ├── buffer.h
│   │   ├── unique_ptr.h
│   │   └── memory_pool.h
│   ├── algo/
│   │   ├── kernel_launcher.h      # NEW
│   │   └── cublas_handle.h         # NEW
│   ├── device/
│   │   ├── error.h
│   │   ├── device_utils.h
│   │   └── reduce_kernels.h
│   └── (other device kernels)
├── image/
│   └── filters.h                 # MODIFIED: device-only API
├── matrix/
│   └── ops.h                    # MODIFIED: device-only API
├── parallel/
│   └── primitives.h             # MODIFIED: device-only API
└── convolution/
    └── conv2d.h                 # MODIFIED: device-only API

src/
├── cuda/
│   ├── algo/
│   │   ├── kernel_launcher.cu
│   │   └── cublas_handle.cpp
│   ├── device/
│   │   └── (kernel implementations)
├── image/
│   └── filters.cu
├── matrix/
│   └── ops.cu
├── parallel/
│   └── primitives.cu
└── memory/
    └── (implementations)

tests/
├── kernel_launcher_test.cpp      # NEW
├── cublas_handle_test.cpp        # NEW
├── image_filters_test.cpp        # UPDATED
├── matrix_ops_test.cpp           # UPDATED
└── parallel_test.cpp            # UPDATED
```

## 9. Usage Example

```cpp
#include "cuda/memory/buffer.h"
#include "cuda/image/filters.h"
#include "cuda/matrix/ops.h"

int main() {
    // Create device buffers
    cuda::memory::Buffer<uint8_t> input(1920 * 1080 * 3);
    cuda::memory::Buffer<uint8_t> output(1920 * 1080 * 3);

    // Upload data
    std::vector<uint8_t> host_data(1920 * 1080 * 3);
    input.copy_from(host_data.data(), host_data.size());

    // Process
    cuda::image::brightness(input, output, 1920, 1080, {1.2f, 30.0f});
    cuda::matrix::multiply(matrix_a, matrix_b, result, m, n, k);

    // Download data
    std::vector<uint8_t> result_host(1920 * 1080 * 3);
    output.copy_to(result_host.data(), result_host.size());
}
```

## 10. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] KernelLauncher implementation + tests
- [ ] CublasHandle implementation + tests
- [ ] Buffer already exists

### Phase 2: Matrix Module
- [ ] Implement add, scale, transpose
- [ ] Implement multiply with CublasHandle
- [ ] Tests

### Phase 3: Image Module
- [ ] Refactor brightness (remove raw kernel)
- [ ] Implement sobel
- [ ] Implement gaussian_blur
- [ ] Tests

### Phase 4: Parallel Module
- [ ] Update scan, sort, histogram signatures
- [ ] Tests

### Phase 5: Convolution
- [ ] Implement conv2d
- [ ] Tests

## 11. Breaking Changes (No Backward Compatibility)

- Remove all functions that take `T* host_pointer` parameters
- Users must manage Buffer lifecycle explicitly
- No more automatic H2D/D2D/D2H copying in library functions
- Library is zero-copy from user perspective
