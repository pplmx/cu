---
phase: 42
phase_name: Coverage Infrastructure
status: planning
created: 2026-04-26
requirements:
  - COVR-01
  - COVR-02
  - COVR-03
---

# Phase 42: Coverage Infrastructure - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning
**Mode:** Autonomous (from ROADMAP.md)

## Phase Boundary

Generate detailed coverage reports with line/branch coverage and gap analysis.

## Implementation Decisions

### Approach

Use LLVM source-based coverage (Clang's -fprofile-instr-generate -fcoverage-mapping) with lcov/genhtml for HTML report generation.

### Coverage Targets

1. **Line Coverage** - Every line executed
2. **Branch Coverage** - Every branch taken
3. **Function Coverage** - Every function called

### Per-Module Breakdown

Group coverage by include/nova/ subdirectory:

- MEMORY
- ALGO
- NEURAL
- FFT
- GRAPH
- etc.

## Specific Ideas

### COVR-01: HTML Coverage Report

- CMake option: NOVA_COVERAGE=ON
- Build with coverage instrumentation
- Generate HTML via lcov/genhtml

### COVR-02: Coverage Gap Analysis

- Compare against TESTED annotation
- Show untested functions

### COVR-03: Per-Module Breakdown

- Group by module
- Show percentages per module

---

## Context generated for Phase 42: Coverage Infrastructure
