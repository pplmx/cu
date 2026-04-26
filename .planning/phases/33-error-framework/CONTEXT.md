# Phase 33: Error Message Framework

**Milestone:** v1.8 Developer Experience
**Status:** Discussing approach

## Goal

Developers can understand and recover from CUDA errors quickly

## Requirements

| ID | Description |
|----|-------------|
| ERR-01 | Descriptive error messages with CUDA function name, file:line, device context |
| ERR-02 | Error-category-specific recovery hints |
| ERR-03 | cuBLAS status code translation to readable names |
| ERR-04 | std::error_code integration for idiomatic error handling |

## Success Criteria

1. Developer sees CUDA error message that includes the CUDA function name, file:line location, and device context (e.g., "Reduce kernel failed at device 0, stream 1, file:line /path/to/kernel.cu:42")
2. Developer sees error-category-specific recovery hints in error output (e.g., "Try reducing block size" for shared memory errors, "Check memory allocation" for OOM errors)
3. Developer sees cuBLAS status codes translated to human-readable names (e.g., `CUBLAS_STATUS_NOT_SUPPORTED` instead of numeric code `7`)
4. Developer can catch and handle Nova errors using `std::error_code` and `std::error_category` idioms in C++

## Existing Codebase

### Error Handling Location
- **Primary:** `include/cuda/device/error.h` — current error exceptions and macros
- **Memory Errors:** `include/cuda/memory_error/memory_error_handler.h` — memory-specific error handling

### Current State (in `include/cuda/device/error.h`)

```cpp
class CudaException : public std::runtime_error {
    explicit CudaException(cudaError_t err, const char* file, int line)
        : std::runtime_error(format_error(err, file, line)), error_(err) {}
    // Only shows: "file:line - CUDA error: <error string>"
};

class CublasException : public std::runtime_error {
    explicit CublasException(cublasStatus_t status, const char* file, int line)
        : std::runtime_error(format_error(status, file, line)), status_(status) {}
    // Only shows: "file:line - cuBLAS error: <numeric code>"
};

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; \
         if (err != cudaSuccess) throw CudaException(err, __FILE__, __LINE__); \
    } while (0)
```

### Current Usage
- `CUDA_CHECK` is used **201 times** across the codebase in `.cu` files
- `CUBLAS_CHECK` is used in `src/cuda/device/cublas_context.cu`

### Gaps Identified

1. **No operation context** — `CUDA_CHECK(cudaMalloc(&ptr, size))` only shows file:line, not which operation failed
2. **No device context** — No device ID, stream, or memory info in errors
3. **cuBLAS shows numeric codes** — `status_` is printed as integer, not `CUBLAS_STATUS_NOT_SUPPORTED`
4. **No std::error_code** — Exceptions only, no idiomatic C++ error handling

## Research Summary

From `.planning/research/SUMMARY.md`:

- Error handling should use `std::error_code` with custom `error_category`
- Recovery hints map error types to suggestions (e.g., `cudaErrorMemoryAllocation` → "Try reducing batch size or freeing memory")
- cuBLAS status codes need exhaustive mapping
- RAII guard pattern (`cuda_error_guard`) for scoped error capture

## Key Questions

1. Should we extend existing `error.h` or create a new error module?
2. What granularity of error categories (per-layer or unified)?
3. Should recovery hints be compile-time strings or runtime-configurable?
4. How to handle async kernel errors (cudaGetLastError)?

---
*Context created: 2026-04-26*
*Ready for planning*
