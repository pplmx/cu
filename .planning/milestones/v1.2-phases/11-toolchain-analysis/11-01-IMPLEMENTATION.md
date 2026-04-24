# Implementation Plan: Phase 12 — Toolchain Upgrade

**Phase:** 12
**Created:** 2026-04-24
**Status:** Ready for Execution

## Overview

This document details the concrete changes required to upgrade the toolchain:
- C++20 → C++23
- CUDA 17 → CUDA 20
- CMake 3.25 → 4.0+

## Required Changes

### 1. Update CMakeLists.txt

**File:** `CMakeLists.txt`

```diff
- cmake_minimum_required(VERSION 3.25)
+ cmake_minimum_required(VERSION 4.0)
```

```diff
- set(CMAKE_CXX_STANDARD 20)
- set(CMAKE_CXX_STANDARD_REQUIRED ON)
- set(CMAKE_CXX_EXTENSIONS OFF)
+ set(CMAKE_CXX_STANDARD 23)
+ set(CMAKE_CXX_STANDARD_REQUIRED ON)
+ set(CMAKE_CXX_EXTENSIONS OFF)
```

```diff
- set(CMAKE_CUDA_STANDARD 17)
- set(CMAKE_CUDA_STANDARD_REQUIRED ON)
- set(CMAKE_CUDA_EXTENSIONS OFF)
+ set(CMAKE_CUDA_STANDARD 20)
+ set(CMAKE_CUDA_STANDARD_REQUIRED ON)
+ set(CMAKE_CUDA_EXTENSIONS OFF)
```

### 2. Update tests/CMakeLists.txt

No changes required — inherits standards from parent project.

### 3. GitHub Actions CI (if applicable)

**File:** `.github/workflows/*.yml`

Check for CUDA version specification:
```yaml
# If using cuda setup action
- uses: awslabs/aws-cuda-toolkit/install@v1
  with:
    cuda: '20'  # Update if applicable
```

### 4. Documentation Updates

**File:** `README.md`
- Update "Requirements" section to reflect new minimum versions

**File:** `CONTRIBUTING.md`
- Update build instructions if necessary

## Rollback Strategy

If issues arise during upgrade:

```bash
# Revert changes
git checkout HEAD~1 -- CMakeLists.txt

# Clean build directory
rm -rf build/
```

## Implementation Order

1. **Create backup branch**
   ```bash
   git checkout -b v1.2-toolchain-upgrade
   ```

2. **Update CMakeLists.txt**
   - Change cmake_minimum_required to 4.0
   - Change CMAKE_CXX_STANDARD to 23
   - Change CMAKE_CUDA_STANDARD to 20

3. **Update CI/CD** (if applicable)
   - Verify CUDA 20 in CI configuration
   - Update Docker base image if applicable

4. **Update documentation**
   - README.md requirements section
   - Any CONTRIBUTING.md build instructions

5. **Build and test**
   ```bash
   rm -rf build/
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ctest --output-on-failure
   ```

6. **Verify all 418 tests pass**

7. **Update planning documents**
   - Mark TC-01 to TC-09 as complete in REQUIREMENTS.md
   - Update STATE.md with completed phases
   - Update ROADMAP.md with Phase 12 complete

## Success Criteria

- [ ] CMakeLists.txt updated with VERSION 4.0, C++23, CUDA 20
- [ ] Build succeeds without errors
- [ ] All 418 tests pass
- [ ] Documentation updated
- [ ] Planning documents updated

## Verification Commands

```bash
# Verify CMake version requirement
grep "cmake_minimum_required" CMakeLists.txt

# Verify C++ standard
grep "CMAKE_CXX_STANDARD" CMakeLists.txt

# Verify CUDA standard
grep "CMAKE_CUDA_STANDARD" CMakeLists.txt

# Build and test
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
ctest --output-on-failure

# Run specific test suites
./bin/nova-tests --gtest_filter="*Test*"
```

---

*Plan created: 2026-04-24*
*Phase 11 analysis complete — ready for Phase 12 execution*
