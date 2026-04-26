# Roadmap — v2.0 Testing & Quality

## Milestone Summary

| Metric | Value |
|--------|-------|
| Milestone | v2.0 Testing & Quality |
| Requirements | 12 |
| Phases | 4 |
| Started | 2026-04-26 |

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 40 | Fuzz Testing Foundation | Build property-based fuzzing infrastructure for CUDA algorithms | FUZZ-01, FUZZ-02, FUZZ-03, FUZZ-04 | 4 criteria |
| 41 | Property-Based Tests | Implement QuickCheck-style tests verifying mathematical and algorithmic properties | PROP-01, PROP-02, PROP-03, PROP-04 | 4 criteria |
| 42 | Coverage Infrastructure | Generate detailed coverage reports with gap analysis | COVR-01, COVR-02, COVR-03 | 3 criteria |
| 43 | CI Integration | Enforce coverage thresholds and integrate all testing in CI pipeline | COVR-04 | 3 criteria |

---

## Phase 40: Fuzz Testing Foundation

**Goal:** Build property-based fuzzing infrastructure for CUDA algorithms using libFuzzer

**Requirements:** FUZZ-01, FUZZ-02, FUZZ-03, FUZZ-04

**Success Criteria:**
1. User can run `make fuzz_memory_pool` and receive pass/fail with corpus count
2. User can run `make fuzz_algorithms` and receive pass/fail with crash count (should be 0)
3. User can run `make fuzz_matmul` and receive pass/fail with edge case discoveries
4. Fuzzing corpus and crash artifacts are isolated in `tests/fuzz/corpus/` and `tests/fuzz/crashes/`

**Dependencies:** None (foundation phase)

**Key Decisions:**
- Use libFuzzer for native C++ fuzzing (already supported by LLVM)
- Target memory pool, algorithm (reduce/scan/sort), and matmul modules first
- Minimum corpus size: 1000 entries before declaring stable

---

## Phase 41: Property-Based Tests

**Goal:** Implement QuickCheck-style tests verifying mathematical and algorithmic properties

**Requirements:** PROP-01, PROP-02, PROP-03, PROP-04

**Success Criteria:**
1. User can run `./tests/property/mathematical` and see all invariants pass
2. User can run `./tests/property/algorithmic` and see all correctness checks pass
3. User can run `./tests/property/numerical` and see stability tests across FP16/FP32/FP64
4. Failed tests output the seed value for exact reproduction

**Dependencies:** Phase 40 (fuzzing infrastructure provides input generation patterns)

**Key Decisions:**
- Use rapidcheck or hand-rolled generator for property tests
- Mathematical invariants: matmul identity (A @ I = A), FFT inverse (FFT^-1(FFT(x)) ≈ x)
- Algorithmic invariants: sort produces sorted output, reduce is associative
- Store seeds in test metadata for reproducibility

---

## Phase 42: Coverage Infrastructure

**Goal:** Generate detailed coverage reports with line/branch coverage and gap analysis

**Requirements:** COVR-01, COVR-02, COVR-03

**Success Criteria:**
1. User can run `make coverage` and receive HTML report at `build/coverage/index.html`
2. User can view coverage gaps via `make coverage-gaps` showing untested functions
3. User can view per-module breakdown showing MEMORY: 85%, ALGO: 78%, NEURAL: 72%, etc.

**Dependencies:** Phase 40 and 41 (coverage requires compiled tests)

**Key Decisions:**
- Use lcov/genhtml for coverage report generation
- LLVM source-based coverage ( `-fprofile-instr-generate -fcoverage-mapping`)
- Gap analysis compares coverage to TESTED annotation in headers
- Per-module breakdown groups by include/nova/ subdirectory

---

## Phase 43: CI Integration

**Goal:** Enforce coverage thresholds and integrate all testing in CI pipeline

**Requirements:** COVR-04

**Success Criteria:**
1. PR fails if `make coverage` reports line coverage below 80%
2. PR fails if fuzzing corpus drops below 90% of baseline corpus size
3. All v2.0 test suites run in parallel on PR (fuzz + property + coverage in ~10 min)

**Dependencies:** Phase 40, 41, 42 (all tests must exist before CI integration)

**Key Decisions:**
- Coverage gate: `MIN_COVERAGE=80` in CI environment variable
- Baseline corpus stored in repository at `tests/fuzz/baseline/`
- Parallel execution via CMake CTest resource files
- Timeout per fuzz target: 5 minutes (balance depth vs CI time)

---

## Requirement Coverage Matrix

| REQ-ID | Phase | Covered |
|--------|-------|---------|
| FUZZ-01 | 40 | ✅ |
| FUZZ-02 | 40 | ✅ |
| FUZZ-03 | 40 | ✅ |
| FUZZ-04 | 40 | ✅ |
| PROP-01 | 41 | ✅ |
| PROP-02 | 41 | ✅ |
| PROP-03 | 41 | ✅ |
| PROP-04 | 41 | ✅ |
| COVR-01 | 42 | ✅ |
| COVR-02 | 42 | ✅ |
| COVR-03 | 42 | ✅ |
| COVR-04 | 43 | ✅ |

**All 12 requirements mapped across 4 phases.**

---

## Execution Order

```
Phase 40 (Fuzz Testing Foundation)
         ↓
Phase 41 (Property-Based Tests)  ← can start after Phase 40 partial
         ↓
Phase 42 (Coverage Infrastructure) ← requires Phases 40+41
         ↓
Phase 43 (CI Integration) ← final integration phase
```

---
*Roadmap created: 2026-04-26 for v2.0 Testing & Quality*
*4 phases, 12 requirements, all covered*
