---
phase: "01"
plan_id: "01-01"
plan_name: "CUDA Context Fixes for Test Fixtures"
wave: 1
autonomous: true

objective: Add cudaSetDevice(0) to all test fixtures that currently skip with CUDA context issues, remove GTEST_SKIP(), and verify tests pass.

files_modified:
  - tests/sparse/preconditioner_test.cpp
  - tests/sparse/reordering_test.cpp
  - tests/sparse/preconditioned_solver_test.cpp
  - tests/algo/sssp_test.cpp
  - tests/cuda/production/memory_node_test.cpp
  - tests/cuda/production/graph_executor_test.cpp

tasks:
  - id: "1"
    description: "Fix JacobiPreconditionerTest fixture"
    files:
      - tests/sparse/preconditioner_test.cpp
    verification: "GTEST_SKIP removed, cudaSetDevice(0) added to SetUp(), tests pass"

  - id: "2"
    description: "Fix RCMReordererTest fixture"
    files:
      - tests/sparse/reordering_test.cpp
    verification: "GTEST_SKIP removed, cudaSetDevice(0) added to SetUp(), tests pass"

  - id: "3"
    description: "Fix PreconditionedSolverTest fixture"
    files:
      - tests/sparse/preconditioned_solver_test.cpp
    verification: "GTEST_SKIP removed, cudaSetDevice(0) added to SetUp(), tests pass"

  - id: "4"
    description: "Fix SSSPTest fixture"
    files:
      - tests/algo/sssp_test.cpp
    verification: "GTEST_SKIP removed, cudaSetDevice(0) added to SetUp(), tests pass"

  - id: "5"
    description: "Fix MemoryNodeTest fixture"
    files:
      - tests/cuda/production/memory_node_test.cpp
    verification: "GTEST_SKIP removed, cudaSetDevice(0) added to SetUp(), tests pass"

  - id: "6"
    description: "Fix GraphExecutorTest fixture"
    files:
      - tests/cuda/production/graph_executor_test.cpp
    verification: "GTEST_SKIP removed, cudaSetDevice(0) added to SetUp(), tests pass"

key-files:
  created: []
  modified:
    - tests/sparse/preconditioner_test.cpp
    - tests/sparse/reordering_test.cpp
    - tests/sparse/preconditioned_solver_test.cpp
    - tests/algo/sssp_test.cpp
    - tests/cuda/production/memory_node_test.cpp
    - tests/cuda/production/graph_executor_test.cpp
---

# Plan 01-01: CUDA Context Fixes for Test Fixtures

## Objective

Add `cudaSetDevice(0)` to all test fixtures that currently skip with "CUDA context issues" message. Remove GTEST_SKIP() and ensure proper CUDA device initialization in SetUp().

## Tasks

### Task 1: Fix JacobiPreconditionerTest fixture
- **File:** tests/sparse/preconditioner_test.cpp
- **Change:** Replace `GTEST_SKIP() << "JacobiPreconditioner tests have CUDA context issues - skipping"` with:
  ```cpp
  void SetUp() override {
      cudaSetDevice(0);
      cudaDeviceSynchronize();
  }
  ```
- **Verification:** Test suite runs without skips for this fixture

### Task 2: Fix RCMReordererTest fixture
- **File:** tests/sparse/reordering_test.cpp
- **Change:** Replace `GTEST_SKIP() << "RCMReorderer tests have CUDA context issues - skipping"` with:
  ```cpp
  void SetUp() override {
      cudaSetDevice(0);
      cudaDeviceSynchronize();
  }
  ```
- **Verification:** Test suite runs without skips for this fixture

### Task 3: Fix PreconditionedSolverTest fixture
- **File:** tests/sparse/preconditioned_solver_test.cpp
- **Change:** Replace `GTEST_SKIP() << "PreconditionedSolver tests have CUDA context issues - skipping"` with:
  ```cpp
  void SetUp() override {
      cudaSetDevice(0);
      cudaDeviceSynchronize();
  }
  ```
- **Verification:** Test suite runs without skips for this fixture

### Task 4: Fix SSSPTest fixture
- **File:** tests/algo/sssp_test.cpp
- **Change:** Replace `GTEST_SKIP() << "SSSPTest has CUDA context issues - skipping"` with:
  ```cpp
  void SetUp() override {
      cudaSetDevice(0);
      cudaDeviceSynchronize();
  }
  ```
- **Verification:** Test suite runs without skips for this fixture

### Task 5: Fix MemoryNodeTest fixture
- **File:** tests/cuda/production/memory_node_test.cpp
- **Change:** Replace `GTEST_SKIP() << "MemoryNodeTest has CUDA context issues - skipping"` or add SetUp with cudaSetDevice(0)
- **Verification:** Test suite runs without skips for this fixture

### Task 6: Fix GraphExecutorTest fixture
- **File:** tests/cuda/production/graph_executor_test.cpp
- **Change:** Replace `GTEST_SKIP() << "GraphExecutorTest has CUDA context issues - skipping"` or add SetUp with cudaSetDevice(0)
- **Verification:** Test suite runs without skips for this fixture

## Success Criteria

1. All 6 test fixtures no longer skip
2. cudaSetDevice(0) added to each SetUp() method
3. Tests run to completion without SEGFAULT
4. 45 total tests pass (11+13+6+4+8+3)
