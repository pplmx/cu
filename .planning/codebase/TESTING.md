# Testing

**Mapped:** 2026-04-23

## Framework

**Google Test** (gtest) v1.14.0, fetched via CMake FetchContent

## Test Organization

| Location | Pattern | Count |
|----------|---------|-------|
| `tests/*_test.cu` | CUDA device tests | 13 test files |
| `tests/*_test.cpp` | C++ host tests | 6 test files |
| `tests/unit/` | Unit test directory | (subdirectory) |
| `tests/integration/` | Integration tests | (subdirectory) |

## Test Structure

### CUDA Tests (*_test.cu)

```cpp
#include <gtest/gtest.h>
#include "cuda/memory/buffer.h"

class ReduceTest : public ::testing::Test {
protected:
    size_t size_ = 1024;
    std::vector<int> h_input_;
    cuda::memory::Buffer<int> d_input_;

    void SetUp() override {
        h_input_.resize(size_);
        d_input_ = cuda::memory::Buffer<int>(size_);
    }

    void TearDown() override {
        d_input_.release();
    }
};

TEST_F(ReduceTest, SumBasic) {
    // Test implementation
    EXPECT_EQ(result, expected);
}
```

### C++ Tests (*_test.cpp)

Standard Google Test patterns for host-side code:

```cpp
#include <gtest/gtest.h>

class MemoryPoolTest : public ::testing::Test {
    // ...
};
```

## Test Suites

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| **ReduceTest** | 11 | reduce_sum, reduce_max, reduce_min |
| **ScanTest** | 10 | Prefix sum operations |
| **SortTest** | 7 | Sorting algorithms |
| **OddEvenSortTest** | 3 | Odd-even transposition sort |
| **MatrixMultTest** | 7 | Matrix multiplication |
| **MatrixOpsTest** | 16 | Matrix operations |
| **ImageBufferTest** | 5 | Image buffer operations |
| **GaussianBlurTest** | 7 | Gaussian blur filter |
| **SobelTest** | 7 | Sobel edge detection |
| **BrightnessTest** | 10 | Brightness adjustments |
| **TestPatternsTest** | 14 | Testing pattern validation |

**Total:** 81+ tests across 13 test suites

## Running Tests

```bash
make test          # Run all tests
make test-unit     # Run algorithm tests only (nova-tests)
make test-patterns # Run pattern tests (test_patterns-tests)
```

## Test Executables

| Executable | Source |
|------------|--------|
| `nova-tests` | Built from `tests/*_test.cu` |
| `test_patterns-tests` | Built from `tests/test_patterns_test.cpp` |

## Build Integration

CMake automatically discovers and builds tests via `enable_testing()` and `add_subdirectory(tests)`:

```cmake
enable_testing()
add_subdirectory(tests)
```

## CI Integration

GitHub Actions runs tests on every push via `.github/workflows/ci.yml`.
