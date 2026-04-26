# Roadmap: Nova CUDA Library — v1.7 Benchmarking & Testing

**Milestone:** v1.7 Benchmarking & Testing
**Created:** 2026-04-26
**Total Phases:** 4 | **Total Requirements:** 27

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 29 | Benchmark Infrastructure Foundation | Stable measurement infrastructure with CUDA event timing, warmup protocols, and NVTX framework | BENCH-01 to BENCH-05 | 4 |
| 30 | Comprehensive Benchmark Suite | Algorithmic benchmarks covering reduce, scan, sort, FFT, matmul, memory ops, and multi-GPU NCCL collectives | SUITE-01 to SUITE-09 | 6 |
| 31 | CI Regression Testing | GitHub Actions workflow with statistical baseline comparison and regression gating | CI-01 to CI-07 | 4 |
| 32 | Performance Dashboards | HTML dashboard with trend charts, baseline comparison, and regression visualization | DASH-01 to DASH-06 | 4 |

---

## Phase 29: Benchmark Infrastructure Foundation

**Goal:** Establish stable, accurate measurement methodology that all downstream benchmarking depends on.

**Requirements:** BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05

**Success Criteria:**

1. Developer can invoke the full benchmark suite with `python scripts/benchmark/run_benchmarks.py --all` and receive structured output
2. Benchmark timing measurements match expected values for known-duration kernels (verified with cudaEvent synchronization)
3. First-run vs. steady-state benchmark times differ significantly, confirming warmup is effective
4. NVTX annotations are absent from timing data when the annotation toggle is disabled (verified by comparing run times with/without NVTX)
5. NVTX timeline annotations appear correctly in Nsight Systems when enabled (verified by inspecting generated `.nsys-rep` file)

**Implementation Notes:**

- Build on existing `cuda::benchmark::Benchmark` class patterns from the codebase
- Add CUDA event-based timing wrapper compatible with Google Benchmark's `UseManualTime()`
- Implement GPU clock stabilization (fixed frequency via `nvidia-smi -lgc` or warmup iteration discard)
- Create `include/cuda/benchmark/nvtx.h` with RAII scoped_range guards
- Separate timing code from NVTX annotation code to prevent overhead distortion

---

## Phase 30: Comprehensive Benchmark Suite

**Goal:** Cover all major algorithm categories with parameterized benchmarks across production-scale input sizes.

**Requirements:** SUITE-01, SUITE-02, SUITE-03, SUITE-04, SUITE-05, SUITE-06, SUITE-07, SUITE-08, SUITE-09

**Success Criteria:**

1. Each algorithm (reduce, scan, sort, FFT, matmul) reports throughput in GB/s that matches published cuBLAS/cuFFT reference values within 10%
2. Multi-GPU NCCL benchmarks show scaling efficiency >80% when adding GPUs (verified on 2+ GPU topology)
3. Benchmark suite completes in <30 minutes on a single A100 for the standard configuration
4. Parameterized benchmarks produce scaling curves across at least 3 orders of magnitude in input size
5. All benchmark results are available as JSON with fields: `name`, `real_time`, `cpu_time`, `bytes_per_second`, `items_per_second`, `iterations`, `threads`
6. Multi-GPU benchmarks aggregate timing correctly across ranks (no peer-to-peer synchronization errors)

**Implementation Notes:**

- Create `benchmark/` directory with C++ Google Benchmark source files
- Implement benchmarks using the patterns established in Phase 29
- Add `benchmark/configs/standard.json` for reproducible workload configurations
- Benchmark all five layers: memory (H2D/D2H/D2D), algo (reduce/scan/sort/FFT), api (matmul), distributed (NCCL collectives)
- Input size ranges should cover: 1KB to 256MB for memory-bandwidth-limited ops, 1K to 128M elements for compute-bound ops
- For multi-GPU: use existing DeviceMesh from v1.1 to detect topology, aggregate timing across ranks

---

## Phase 31: CI Regression Testing

**Goal:** Automated regression detection in CI with statistical rigor and actionable failure output.

**Requirements:** CI-01, CI-02, CI-03, CI-04, CI-05, CI-06, CI-07

**Success Criteria:**

1. GitHub Actions workflow runs on PR and posts benchmark results as PR comment or CI artifact
2. Workflow fails (blocks merge) when any benchmark regresses beyond configured tolerance with statistical significance (p < 0.01)
3. Baseline JSON files in `scripts/benchmark/baselines/` include metadata: `{ "commit": "...", "gpu": "...", "cuda": "...", "driver": "...", "date": "..." }`
4. `scripts/benchmark/check_regression.py` produces output: `"Benchmark X: ±Y% (current: Zms, baseline: Wms, threshold: T%)"` on both pass and fail
5. CI workflow detects and reports when baselines are older than 30 days
6. CI regression failures are reproducible locally (same command that CI runs produces the same result)

**Implementation Notes:**

- Create `scripts/benchmark/run_benchmarks.py` with `--all`, `--filter`, `--config` arguments
- Create `scripts/benchmark/check_regression.py` using `scipy.stats.ttest_ind` for Welch's t-test
- Store baselines in git under `scripts/benchmark/baselines/` with version tags (e.g., `v1.7.0/`)
- GitHub Actions workflow: trigger on PR, run benchmarks, compare against baselines, fail if regression exceeds threshold
- Use `BENCHMARK_ENABLE_GTEST_TESTS=OFF` in CMake to avoid conflicts with existing Google Test
- Tolerance defaults: 5% for memory ops, 10% for compute ops, 15% for distributed ops

---

## Phase 32: Performance Dashboards

**Goal:** Visual performance reporting that makes trends and regressions immediately visible.

**Requirements:** DASH-01, DASH-02, DASH-03, DASH-04, DASH-05, DASH-06

**Success Criteria:**

1. Running `python scripts/benchmark/generate_dashboard.py --results results/latest --output reports/` produces `reports/index.html`
2. Dashboard table shows all benchmarks with columns: Name, Time (ms), Throughput, Change vs Baseline, Status
3. Trend chart shows at least the last 5 benchmark runs with regression annotations
4. Regressions display in red (#ef4444), improvements in green (#22c55e), stable results in gray
5. Dashboard footer includes GPU model, CUDA version, driver version, and benchmark run timestamp
6. Generated HTML is self-contained with no external dependencies (CSS and JS inlined or bundled)

**Implementation Notes:**

- Create `scripts/benchmark/generate_dashboard.py` using `pandas` for data processing and `plotly` for charts
- Use `jinja2` or `chevron` for HTML template rendering
- Dashboard template at `scripts/benchmark/templates/dashboard.html` with inline CSS/JS
- Plotly HTML output is self-contained by default (`plotly.io.write_html` with `full_html=True`)
- CI can attach `reports/` as GitHub Actions artifact or push to GitHub Pages
- Include a "Compare" view that overlays current run against selected baseline version

---

## Phase Ordering Rationale

| Phase | Reason for Order |
|-------|------------------|
| Phase 29 first | Measurement methodology underpins everything. Without stable timing, benchmarks produce misleading data. |
| Phase 30 second | Core algorithmic benchmarks are the reason the infrastructure exists. |
| Phase 31 third | CI gates need benchmarks to run and baselines to compare against. Statistical rigor depends on Phase 29 infrastructure. |
| Phase 32 last | Dashboards consume data from Phases 30-31. No point in visualizing data that isn't reliable yet. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BENCH-01 | Phase 29 | Pending |
| BENCH-02 | Phase 29 | Pending |
| BENCH-03 | Phase 29 | Pending |
| BENCH-04 | Phase 29 | Pending |
| BENCH-05 | Phase 29 | Pending |
| SUITE-01 | Phase 30 | Pending |
| SUITE-02 | Phase 30 | Pending |
| SUITE-03 | Phase 30 | Pending |
| SUITE-04 | Phase 30 | Pending |
| SUITE-05 | Phase 30 | Pending |
| SUITE-06 | Phase 30 | Pending |
| SUITE-07 | Phase 30 | Pending |
| SUITE-08 | Phase 30 | Pending |
| SUITE-09 | Phase 30 | Pending |
| CI-01 | Phase 31 | Pending |
| CI-02 | Phase 31 | Pending |
| CI-03 | Phase 31 | Pending |
| CI-04 | Phase 31 | Pending |
| CI-05 | Phase 31 | Pending |
| CI-06 | Phase 31 | Pending |
| CI-07 | Phase 31 | Pending |
| DASH-01 | Phase 32 | Pending |
| DASH-02 | Phase 32 | Pending |
| DASH-03 | Phase 32 | Pending |
| DASH-04 | Phase 32 | Pending |
| DASH-05 | Phase 32 | Pending |
| DASH-06 | Phase 32 | Pending |

**Coverage:**
- v1.7 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0 ✓

---
*Roadmap created: 2026-04-26*
*4 phases | 27 requirements | Ready to execute*
