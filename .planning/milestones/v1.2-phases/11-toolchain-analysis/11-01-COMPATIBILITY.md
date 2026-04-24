# Compatibility Report: Phase 11 — Toolchain Analysis

**Phase:** 11
**Created:** 2026-04-24
**Status:** Complete

## Executive Summary

This report documents the analysis of upgrading the Nova CUDA library toolchain:
- C++ Standard: 20 → 23
- CUDA Standard: 17 → 20
- CMake Version: 3.25 → 4.0+

**Overall Assessment:** ✅ **No breaking changes expected.** The codebase uses standard C++ and CUDA features that are fully compatible with the newer versions.

## C++ Standard Compatibility Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| `std::vector`, `std::unique_ptr`, `std::shared_ptr` | ✅ Compatible | C++98+, always available |
| `std::optional`, `std::variant` | ✅ Compatible | C++17+, available in C++23 |
| `std::string_view` | ✅ Compatible | C++17+, available in C++23 |
| `std::expected` | ℹ️ New in C++23 | Candidate for future error handling |
| `std::print` | ℹ️ New in C++23 | Formatted output (not currently used) |
| Lambda expressions | ✅ Compatible | C++11+, fully supported |
| `std::clamp` | ✅ Compatible | C++17+, used in stream_manager.h:56 |
| `std::abs` (integer) | ✅ Compatible | C++11+, used in benchmark.h |
| Move semantics | ✅ Compatible | C++11+, used throughout |
| `[[nodiscard]]` | ✅ Compatible | C++17+, used in benchmark.h, event.h, stream.h |
| Template concepts | ✅ Compatible | C++20+, not currently used |

### C++20 Features Used

The codebase currently uses **no C++20-specific features** that would break under C++23:
- No concepts/requires clauses
- No coroutines (co_await, co_return, co_yield)
- No consteval
- Standard library is C++17-compatible

### C++23 Considerations

**Backward Compatible:** All C++20 code will compile under C++23 without changes.

**Optional Enhancements for Future:**
1. `std::expected<T, E>` — Replace `std::variant<T, std::error_code>` pattern
2. `std::to_underlying` — Cleaner enum casting
3. `std::views::chunk_by` — Range grouping
4. `import std;` — Module import (requires build system support)

## CUDA Standard Compatibility Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| `__global__` kernels | ✅ Compatible | Core CUDA syntax |
| `__launch_bounds__` | ✅ Compatible | Used throughout |
| `__restrict__` pointers | ✅ Compatible | Used in distributed/reduce.cu |
| `cudaStream_t`, `cudaEvent_t` | ✅ Compatible | Standard CUDA types |
| Template kernels | ✅ Compatible | Used in reduce_kernels.cu |
| Device lambdas | ✅ Compatible | Not currently used |
| `__CUDA_ARCH__` | ✅ Compatible | Used for compute capability checks |

### CUDA Version-Specific Analysis

**Current CUDA 17 Usage:**
- Standard kernel launches (`<<< >>>`)
- Device/host code separation
- cuBLAS integration (`cuda::blas::Context`)
- cuFFT integration (`cuda::fft::FFTPlan`)
- Unified memory operations

**CUDA 20 Compatibility:**
- All CUDA 17 APIs are expected to remain available
- No deprecated APIs detected in current codebase
- Device code follows standard patterns

### Architecture Support

Current: `60 70 80 90` (Kepler through Ampere)

**CUDA 20 Consideration:**
- CUDA 20 may add support for compute capability 10.x (if new hardware exists)
- Current architectures should remain supported
- No changes expected to `CMAKE_CUDA_ARCHITECTURES`

## CMake Version Compatibility Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| `cmake_minimum_required(VERSION 3.25)` | ✅ Compatible | Will bump to 4.0 |
| `FetchContent_MakeAvailable` | ✅ Compatible | CMake 3.24+ |
| Generator expressions (`$<1:...>`) | ✅ Compatible | Standard usage |
| `find_package(CUDAToolkit)` | ✅ Compatible | Standard CMake |
| `target_link_libraries` | ✅ Compatible | Standard CMake |
| `set_target_properties` | ✅ Compatible | Standard CMake |
| `CXX_STANDARD` properties | ✅ Compatible | Standard CMake |
| `CUDA_SEPARABLE_COMPILATION` | ✅ Compatible | CUDA-specific property |

### CMake 4.0+ Breaking Changes (to verify)

| Change | Impact | Mitigation |
|--------|--------|------------|
| Policy CMP0148 (if any) | Low | Test in staging |
| `GENERATE` property syntax | Low | Not currently used |
| Module deprecations | Low | Check cmake/FindXXX.cmake |

### CMake 4.0+ Opportunities

1. **Better dependency management** with updated FetchContent
2. **Improved CUDA integration** in CMake itself
3. **Potential for `cmake_language` enhancements**

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| C++23 breaking changes | Low | Very Low | C++23 is backward compatible |
| CUDA 20 API deprecation | Medium | Low | Check NVIDIA release notes |
| CMake 4.0 policy changes | Low | Low | Test incremental upgrades |
| Test failures after upgrade | Medium | Low | Run full test suite |

## Recommendations

1. **Bump CMake minimum to 4.0** — Low risk, enables new features
2. **Bump C++ standard to 23** — Zero breaking changes, enables future features
3. **Bump CUDA standard to 20** — Verify CUDA 20 availability in CI
4. **Add `std::expected` pattern** — Future enhancement (Phase 13+)
5. **Update CI/CD** — Ensure CUDA 20 toolchain available in CI runners

## Verification Plan

### Before Upgrade
- [ ] Check CUDA 20 availability in CI
- [ ] Review CUDA 20 release notes for breaking changes
- [ ] Verify CMake 4.0+ available in development environments

### After Upgrade
- [ ] Run full test suite (418 tests)
- [ ] Verify no compiler warnings
- [ ] Check performance regression (benchmark suite)
- [ ] Validate CUDA architectures work correctly

---

*Report generated: 2026-04-24*
*Next action: Proceed to Phase 12 — Toolchain Upgrade*
