# Phase 96: Dashboard & Visualization — Context

**Created:** 2026-05-02
**Milestone:** v2.11 Performance Tooling

## Requirements

- **DASH-01:** Extended performance dashboard with fusion impact visualization
- **DASH-02:** Roofline plot export (JSON/CSV) for external tools
- **DASH-03:** Kernel-level flame graph generation from NVTX traces
- **INT-01:** Integration with existing nvbench_integration.h
- **INT-02:** NVTX domain extensions for new profiling categories

## Prior Work

### Phase 93: NVBlox Foundation
- NVBloxMetricsCollector for kernel metrics
- KernelProfiler with CUDA event tracking

### Phase 94: Kernel Fusion Analysis
- KernelFusionAnalyzer for pattern detection
- FusionRecommendation with confidence levels

### Phase 95: Memory Bandwidth
- RooflineModel for operational intensity
- BandwidthUtilizationTracker

### Existing Infrastructure

#### generate_dashboard.py
- HTML dashboard with Plotly charts
- Regression tracking and comparison
- Environment context display

## Implementation Strategy

### DashboardDataExporter
```cpp
class DashboardDataExporter {
    void export_roofline_data(const RooflineModel& model);
    void export_fusion_recommendations(const std::vector<FusionRecommendation>& recs);
    std::string generate_dashboard_json() const;
};
```

### FlameGraphGenerator
- Parse NVTX Chrome trace data
- Generate flame graph JSON format
- Hierarchical kernel timing

### NVTX Extensions
Added in Phase 93:
- `nova.performance` - General performance
- `nova.performance.nvblox` - NVBlox metrics
- `nova.performance.fusion` - Fusion analysis
- `nova.performance.bandwidth` - Bandwidth analysis
