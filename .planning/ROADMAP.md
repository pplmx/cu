# Roadmap — v2.11 Performance Tooling

## Milestone Summary

**Milestone:** v2.11 Performance Tooling
**Goal:** Comprehensive performance tooling — NVBlox integration, kernel fusion analysis, memory bandwidth optimization
**Requirements:** 16 (NVBlox-01 to NVBlox-03, FUSION-01 to FUSION-03, BANDWIDTH-01 to BANDWIDTH-03, DASH-01 to DASH-03, INT-01 to INT-02)
**Phases:** 5

---

## Phase 93: NVBlox Foundation

**Goal:** Integrate NVBlox for kernel-level performance analysis with custom metrics

**Requirements:**
- NVBlox-01: NVBlox metrics integration header with custom metric registration
- NVBlox-02: Kernel-level profiling hooks for latency, throughput, occupancy
- NVBlox-03: Custom metric aggregators (arithmetic intensity, FLOP/s, memory BW)

**Success Criteria:**
1. NVBloxcMetricsCollector class with RAII initialization
2. Custom metric registration API (register_metric, add_sample)
3. Per-kernel latency, throughput, and SM occupancy tracking
4. Arithmetic intensity calculation per kernel
5. FLOP/s and memory bandwidth metric aggregation
6. CMake NVBLOX_FOUND detection with graceful fallback
7. >95% unit test coverage for metric collection

**Implementation Notes:**
- Follow existing nvbench_integration.h patterns from v2.4
- Use nvblox::gpu::GpuMetrics if available, fallback to CUDA events
- Store metrics in std::vector<KernelMetrics> for aggregation
- Thread-safe metric collection via mutex protection

---

## Phase 94: Kernel Fusion Analysis

**Goal:** Detect fusion opportunities and provide profitability-based recommendations

**Requirements:**
- FUSION-01: KernelFusionAnalyzer with op-to-op fusion opportunity detection
- FUSION-02: Fusion profitability model (break-even analysis based on launch overhead)
- FUSION-03: Fusion recommendation engine with confidence levels

**Success Criteria:**
1. KernelFusionAnalyzer scans operation graph for adjacent ops
2. Fusion opportunity detection: matmul+bias+act, relu+pool, etc.
3. Profitability model: launch_overhead_saved vs memory_coalescing_tradeoff
4. Break-even threshold configurable (default: 100us launch overhead)
5. Confidence levels: HIGH (pattern match), MEDIUM (heuristic), LOW (statistical)
6. Fusion recommendation report with before/after cost estimate
7. Unit tests for 10+ known fusion patterns

**Implementation Notes:**
- Build on existing GraphExecutor patterns from v2.4
- Pattern library for common fusions (matmul+act, conv+relu, etc.)
- Configurable profitability thresholds via struct
- Export recommendations as JSON for CI integration

---

## Phase 95: Memory Bandwidth Optimization

**Goal:** Roofline model and comprehensive memory bandwidth analysis

**Requirements:**
- BANDWIDTH-01: RooflineModel class with theoretical peak and achieved performance
- BANDWIDTH-02: Memory bandwidth utilization tracker (H2D/D2H/D2D)
- BANDWIDTH-03: Cache hit rate analysis (L1/L2/texture)

**Success Criteria:**
1. RooflineModel::compute(arithmetic_intensity, flops, bandwidth) returns operational intensity
2. Peak FLOP/s from device properties (FP64/FP32/FP16)
3. Peak bandwidth from device properties (HBM)
4. Bandwidth utilization percentage for each kernel
5. Cache analysis via CUpti events (L1 hit rate, L2 hit rate)
6. Roofline plot data export (JSON with points, lines, axes)
7. Integration with existing BandwidthAnalyzer from v2.7

**Implementation Notes:**
- Extend existing BandwidthAnalyzer from v2.7
- Use cuPTI or CUDA events for cache metrics
- Roofline point: (AI, Performance) per kernel
- Warning if achieved < 50% of roofline (memory bound)

---

## Phase 96: Dashboard & Visualization

**Goal:** Extended performance dashboards with fusion impact and roofline visualization

**Requirements:**
- DASH-01: Extended performance dashboard with fusion impact visualization
- DASH-02: Roofline plot export (JSON/CSV) for external tools
- DASH-03: Kernel-level flame graph generation from NVTX traces
- INT-01: Integration with existing nvbench_integration.h
- INT-02: NVTX domain extensions for new profiling categories

**Success Criteria:**
1. Dashboard shows: kernel timeline, fusion recommendations, roofline plot
2. Fusion impact panel: before/after FLOP/s, memory BW, kernel count
3. Roofline plot with logarithmic axes and operational intensity markers
4. Flame graph generation from Chrome trace NVTX data
5. JSON/CSV export for all metrics (roofline, fusion, bandwidth)
6. New NVTX domain "performance" with sub-domains (nvblox, fusion, bandwidth)
7. Backward compatible with existing v1.7/v2.7 dashboards

**Implementation Notes:**
- Extend existing HTML dashboard generator
- Use Plotly.js for roofline plot (reuse v1.7 pattern)
- Generate flame graph JSON from NVTX ranges
- NVTX domain structure: nvtx::domain("nova::performance")

---

## Phase 97: Integration & Validation

**Goal:** Validate all tooling and finalize milestone

**Requirements:**
- Integration tests
- Performance benchmarks
- Documentation updates

**Success Criteria:**
1. All 16 requirements have passing tests
2. Benchmark suite validates profiling overhead < 5%
3. Dashboard generation time < 2 seconds for 1000-kernel trace
4. Build succeeds with and without NVBlox dependency
5. Documentation: PERFORMANCE_TOOLING.md with examples
6. E2E test: run_benchmarks.py generates full dashboard
7. CI integration: GitHub Actions runs performance tooling tests

**Implementation Notes:**
- Use existing benchmark infrastructure from v1.7
- Test both with and without NVBlox (CMake options)
- Update docs/PRODUCTION.md with tooling section
- Add example: `examples/performance_profiling.cpp`

---

## Phase Map

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 93 | NVBlox Foundation | Kernel-level profiling | NVBlox-01, NVBlox-02, NVBlox-03 | 7 |
| 94 | Kernel Fusion Analysis | Fusion detection & recommendations | FUSION-01, FUSION-02, FUSION-03 | 7 |
| 95 | Memory Bandwidth | Roofline model & cache analysis | BANDWIDTH-01, BANDWIDTH-02, BANDWIDTH-03 | 7 |
| 96 | Dashboard & Visualization | Extended dashboards & flame graphs | DASH-01, DASH-02, DASH-03, INT-01, INT-02 | 7 |
| 97 | Integration & Validation | Finalize and document | Integration tests, benchmarks, docs | 7 |

---

## Traceability Matrix

| REQ | Phase 93 | Phase 94 | Phase 95 | Phase 96 | Phase 97 |
|-----|----------|----------|----------|----------|----------|
| NVBlox-01 | ✓ | | | | |
| NVBlox-02 | ✓ | | | | |
| NVBlox-03 | ✓ | | | | |
| FUSION-01 | | ✓ | | | |
| FUSION-02 | | ✓ | | | |
| FUSION-03 | | ✓ | | | |
| BANDWIDTH-01 | | | ✓ | | |
| BANDWIDTH-02 | | | ✓ | | |
| BANDWIDTH-03 | | | ✓ | | |
| DASH-01 | | | | ✓ | |
| DASH-02 | | | | ✓ | |
| DASH-03 | | | | ✓ | |
| INT-01 | | | | ✓ | |
| INT-02 | | | | ✓ | |

**Coverage:** 14/14 requirements mapped (100%)

---

## Dependencies

- Phase 93: Phase 92 (HealthMetrics, OccupancyAnalyzer)
- Phase 94: Phase 93 (requires metrics data)
- Phase 95: Phase 92 (extends BandwidthAnalyzer)
- Phase 96: Phases 93, 94, 95 (uses all metrics)
- Phase 97: Phases 93-96 (integrates all)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NVBlox API changes | Medium | Graceful fallback to CUDA events |
| Fusion heuristics inaccurate | Medium | Confidence levels, manual override |
| Cache metrics not available | Low | Use alternative metrics (L2 throughput) |

---

*Roadmap created: 2026-05-02*
*5 phases, 14 core requirements + 2 integration requirements*
