# Stage 3: Parallel Algorithms - Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement task-by-task.

**Goal:** 实现三个并行算法(Reduce, Scan, Sort),每个都有基础版和优化版

**Architecture:** 每个算法独立的 header/implementation 文件,使用模板支持多类型

**Tech Stack:** CUDA C++, GoogleTest

---

## Task 1: Reduce (归约)

**Files:**
- Create: `include/reduce.h`
- Create: `src/reduce.cu`
- Create: `tests/reduce_test.cu`
- Modify: `tests/CMakeLists.txt`

### Step 1: Create include/reduce.h

```cpp
#pragma once

#include <cstddef>

enum class ReduceOp { SUM, MAX, MIN };

template<typename T>
T reduceSum(const T* d_input, size_t size);

template<typename T>
T reduceMax(const T* d_input, size_t size, int* maxIndex = nullptr);

template<typename T>
T reduceMin(const T* d_input, size_t size);

template<typename T>
T reduceSumOptimized(const T* d_input, size_t size);
```

### Step 2: Create src/reduce.cu

**基础版** - 相邻配对:
```cpp
#include "reduce.h"
#include "cuda_utils.h"

template<typename T>
__device__ T warpReduce(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__global__ void reduceKernel(const T* input, T* output, size_t size) {
    __shared__ T sdata[256];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = (i < size) ? input[i] : 0;
    sum = warpReduce(sum);

    if (tid % warpSize == 0) sdata[tid / warpSize] = sum;
    __syncthreads();

    if (tid < warpSize) sum = (tid < blockDim.x / warpSize) ? sdata[tid] : 0;
    if (tid < warpSize) sum = warpReduce(sum);

    if (tid == 0) output[blockIdx.x] = sum;
}

template<typename T>
T reduceSum(const T* d_input, size_t size) {
    const int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, gridSize * sizeof(T)));

    reduceKernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    T result = 0;
    for (int i = 0; i < gridSize; ++i) {
        T val;
        CUDA_CHECK(cudaMemcpy(&val, d_output + i, sizeof(T), cudaMemcpyDeviceToHost));
        result += val;
    }

    CUDA_CHECK(cudaFree(d_output));
    return result;
}
```

### Step 3: Create tests/reduce_test.cu

```cpp
#include <gtest/gtest.h>
#include "reduce.h"
#include <numeric>

class ReduceTest : public ::testing::Test {
protected:
    size_t size_ = 1024;
    std::vector<int> h_input_;
    int *d_input_;
};

TEST_F(ReduceTest, SumTest) {
    for (int i = 1; i <= size_; ++i) h_input_[i-1] = i;
    int expected = size_ * (size_ + 1) / 2;
    EXPECT_EQ(reduceSum(h_input_.data(), size_), expected);
}

TEST_F(ReduceTest, MaxTest) {
    h_input_.assign(size_, 0);
    h_input_[500] = 999;
    int result = reduceMax(h_input_.data(), size_);
    EXPECT_EQ(result, 999);
}

TEST_F(ReduceTest, OptimizedMatchesBasic) {
    for (int i = 1; i <= size_; ++i) h_input_[i-1] = i;
    int basic = reduceSum(h_input_.data(), size_);
    int optimized = reduceSumOptimized(h_input_.data(), size_);
    EXPECT_EQ(basic, optimized);
}
```

### Step 4: Update CMakeLists, build, test, commit

---

## Task 2: Scan (前缀和)

**Files:**
- Create: `include/scan.h`
- Create: `src/scan.cu`
- Create: `tests/scan_test.cu`

### Step 1: Create include/scan.h

```cpp
#pragma once

#include <cstddef>

template<typename T>
void exclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void inclusiveScan(const T* d_input, T* d_output, size_t size);

template<typename T>
void exclusiveScanOptimized(const T* d_input, T* d_output, size_t size);
```

### Step 2: Create src/scan.cu

**基础版 (Kogge-Stone)**:
```cpp
#include "scan.h"
#include "cuda_utils.h"

template<typename T>
__global__ void scanKernel(const T* input, T* output, size_t size) {
    __shared__ T temp[1024];
    size_t tid = threadIdx.x;

    if (tid < size) temp[tid] = input[tid];
    else temp[tid] = 0;
    __syncthreads();

    for (int offset = 1; offset < size; offset *= 2) {
        if (tid >= offset) {
            temp[tid] += temp[tid - offset];
        }
        __syncthreads();
    }

    output[tid] = temp[tid];
}

template<typename T>
void exclusiveScan(const T* d_input, T* d_output, size_t size) {
    scanKernel<<<1, size>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

### Step 3: Create tests/scan_test.cu

```cpp
#include <gtest/gtest.h>
#include "scan.h"

class ScanTest : public ::testing::Test {
protected:
    std::vector<int> h_input_, h_output_;
};

TEST_F(ScanTest, PrefixSum) {
    h_input_ = {3, 1, 4, 1, 5, 9, 2, 6};
    exclusiveScan(h_input_.data(), h_output_.data(), h_input_.size());
    std::vector<int> expected = {0, 3, 4, 8, 9, 14, 23, 25};
    EXPECT_EQ(h_output_, expected);
}
```

### Step 4: Build, test, commit

---

## Task 3: Sort (Bitonic Sort)

**Files:**
- Create: `include/sort.h`
- Create: `src/sort.cu`
- Create: `tests/sort_test.cu`

### Step 1: Create include/sort.h

```cpp
#pragma once

#include <cstddef>

template<typename T>
void oddEvenSort(const T* d_input, T* d_output, size_t size);

template<typename T>
void bitonicSort(const T* d_input, T* d_output, size_t size);
```

### Step 2: Create src/sort.cu

**Bitonic Sort**:
```cpp
#include "sort.h"
#include "cuda_utils.h"

template<typename T>
__device__ void compareAndSwap(T& a, T& b, bool ascending) {
    if ((a > b) == ascending) {
        T temp = a;
        a = b;
        b = temp;
    }
}

template<typename T>
__global__ void bitonicSortKernel(T* data, size_t size, int k, int j) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ixj = i ^ j;

    if (ixj > i && ixj < size) {
        bool ascending = ((i & k) == 0);
        if (ascending == (data[i] > data[ixj])) {
            T temp = data[i];
            data[i] = data[ixj];
            data[ixj] = temp;
        }
    }
}

template<typename T>
void bitonicSort(const T* d_input, T* d_output, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_output, d_input, size * sizeof(T), cudaMemcpyHostToDevice));

    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, d_input, size * sizeof(T), cudaMemcpyHostToDevice));

    for (int k = 2; k <= size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortKernel<<<1, size>>>(d_data, size, k, j);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaMemcpy(d_output, d_data, size * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}
```

### Step 3: Create tests/sort_test.cu

```cpp
#include <gtest/gtest.h>
#include "sort.h"
#include <algorithm>

class SortTest : public ::testing::Test {
protected:
    std::vector<int> h_input_, h_output_;
};

TEST_F(SortTest, RandomArray) {
    h_input_ = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    bitonicSort(h_input_.data(), h_output_.data(), h_input_.size());
    std::vector<int> expected = h_input_;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(h_output_, expected);
}
```

### Step 4: Build, test, commit

---

## Task 4: Benchmark & Demo

**Files:**
- Modify: `src/main.cpp`

### Step 1: Update main.cpp

```cpp
#include <iostream>
#include <chrono>
#include "reduce.h"
#include "scan.h"
#include "sort.h"

template<typename Func>
float benchmark(const char* name, Func f, int iterations = 10) {
    float total = 0;
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<float, std::milli>(end - start).count();
    }
    std::cout << name << ": " << (total / iterations) << " ms (avg)" << std::endl;
    return total / iterations;
}

int main() {
    constexpr size_t N = 1 << 20;  // 1M elements

    std::vector<int> input(N);
    std::iota(input.begin(), input.end(), 0);

    int result = 0;
    benchmark("Reduce", [&]() {
        result = reduceSum(input.data(), N);
    });
    std::cout << "  Sum result: " << result << std::endl;

    std::cout << "\nParallel Algorithms Demo Complete!" << std::endl;
    return 0;
}
```

### Step 2: Build, run, commit

---

## Self-Review Checklist

- [ ] Reduce: 基础版+优化版 ✓, sum/max/min ✓
- [ ] Scan: 排他+包含扫描 ✓, Kogge-Stone ✓, Blelloch ✓
- [ ] Sort: Odd-Even ✓, Bitonic ✓
- [ ] GoogleTest 测试 ✓
- [ ] Benchmark ✓

---

**Plan complete.** 执行选项:

1. **Subagent-Driven (推荐)** - 逐任务dispatch
2. **Inline Execution** - 本session执行

选哪个?
