# Phase 95: Memory Bandwidth Optimization — Context

**Created:** 2026-05-02
**Milestone:** v2.11 Performance Tooling

## Requirements

- **BANDWIDTH-01:** RooflineModel class with theoretical peak and achieved performance
- **BANDWIDTH-02:** Memory bandwidth utilization tracker (H2D/D2H/D2D)
- **BANDWIDTH-03:** Cache hit rate analysis (L1/L2/texture)

## Prior Work

### Phase 92: v2.10 Performance Benchmarks
- BandwidthAnalyzer with H2D/D2H/D2D measurement
- DeviceMemoryBandwidth query from device properties

### Existing Infrastructure

#### observability/bandwidth_tracker.h
- `BandwidthTracker` for transfer measurement
- `DeviceMemoryBandwidth` with theoretical peak queries
- `BandwidthResult` with bandwidth, bytes, elapsed time

## Implementation Strategy

### RooflineModel Class
```cpp
class RooflineModel {
public:
    struct RooflinePoint {
        double arithmetic_intensity;
        double performance_gflops;
        std::string kernel_name;
    };
    
    void add_point(const RooflinePoint& point);
    void compute_operational_intensity(uint64_t flops, size_t bytes);
    
    double get_peak_flops_fp64() const;
    double get_peak_bandwidth() const;
    std::string to_json() const;
};
```

### Bandwidth Utilization Tracker
Extend existing BandwidthTracker:
- Track utilization percentage per transfer type
- Historical bandwidth samples
- Warning when utilization < 50%

### Cache Analysis
```cpp
class CacheAnalyzer {
public:
    struct CacheMetrics {
        double l1_hit_rate;
        double l2_hit_rate;
        double texture_hit_rate;
    };
    
    CacheMetrics analyze(int device_id = 0);
};
```

## Dependencies

- Phase 92: BandwidthTracker, DeviceMemoryBandwidth
- Phase 93: NVBloxMetricsCollector, KernelMetrics
- CUDA events for timing
