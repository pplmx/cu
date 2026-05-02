# Requirements — v2.11 Performance Tooling

## Goals & Success Criteria

1. Provide production-ready performance profiling infrastructure via NVBlox integration
2. Enable automated kernel fusion opportunity detection with actionable recommendations
3. Deliver comprehensive memory bandwidth analysis with roofline model visualization
4. Integrate all tooling into existing benchmark/dashboard infrastructure from v1.7/v2.7

## Functional Requirements

### Must Have

- [ ] **NVBlox-01:** NVBlox metrics integration header with custom metric registration
- [ ] **NVBlox-02:** Kernel-level profiling hooks for latency, throughput, occupancy
- [ ] **NVBlox-03:** Custom metric aggregators ( arithmetic intensity, FLOP/s, memory BW)
- [ ] **FUSION-01:** KernelFusionAnalyzer with op-to-op fusion opportunity detection
- [ ] **FUSION-02:** Fusion profitability model (break-even analysis based on launch overhead)
- [ ] **FUSION-03:** Fusion recommendation engine with confidence levels
- [ ] **BANDWIDTH-01:** RooflineModel class with theoretical peak and achieved performance
- [ ] **BANDWIDTH-02:** Memory bandwidth utilization tracker (H2D/D2H/D2D)
- [ ] **BANDWIDTH-03:** Cache hit rate analysis (L1/L2/texture)

### Should Have

- [ ] **DASH-01:** Extended performance dashboard with fusion impact visualization
- [ ] **DASH-02:** Roofline plot export (JSON/CSV) for external tools
- [ ] **DASH-03:** Kernel-level flame graph generation from NVTX traces
- [ ] **INT-01:** Integration with existing nvbench_integration.h from v2.4
- [ ] **INT-02:** NVTX domain extensions for new profiling categories

### Could Have

- [ ] **AUTO-01:** Automated kernel fusion suggestion via ML-based profitability model
- [ ] **AUTO-02:** Performance regression prediction based on historical baselines

## Non-Functional Requirements

### Performance

- Profiling overhead < 5% of kernel execution time
- Dashboard generation < 2 seconds for 1000-kernel trace
- Roofline calculation linear in trace size O(n)

### Reliability

- All new tooling must pass existing benchmark suite
- No regression in existing profiling performance (NVTX, HealthMetrics)
- Graceful degradation when NVBlox unavailable (optional dependency)

### Security

- No external data transmission (all analysis local)
- Safe file paths for dashboard export

### Observability

- All metrics exportable to JSON for CI integration
- NVTX annotations for all new profiling categories
- Error logging for failed metric collection

## Constraints & Assumptions

- Build on existing `nvbench_integration.h`, `HealthMetrics`, and NVTX extensions
- CUDA 20 toolchain assumed (required for NVBlox APIs)
- No breaking changes to existing profiler.h interfaces
- NVBlox is optional — library must build and pass tests without it

## Out of Scope

- Python bindings for profiling tools
- Real-time streaming profiling (trace collection only)
- Multi-node profiling aggregation
- Automated kernel rewrite suggestions

## Dependencies

- Existing: CUDA 20, Google Benchmark, NVTX, nvbench_integration.h
- Optional: NVBlox library (CMake NVBLOX_FOUND option)
- Existing: HealthMetrics, OccupancyAnalyzer from v2.7

## Open Questions

- [ ] Should NVBlox metrics be compile-time optional (NOVA_ENABLE_NVBLOX)?
- [ ] What baseline thresholds for fusion profitability (e.g., >100us launch overhead)?
- [ ] Preferred format for roofline export (JSON for tools, CSV for spreadsheets)?

---

*Requirements created: 2026-05-02*
*Total: 16 requirements (9 Must Have, 5 Should Have, 2 Could Have)*
