# Phase 93: NVBlox Foundation — Context

**Created:** 2026-05-02
**Milestone:** v2.11 Performance Tooling

## Requirements

- **NVBlox-01:** NVBlox metrics integration header with custom metric registration
- **NVBlox-02:** Kernel-level profiling hooks for latency, throughput, occupancy
- **NVBlox-03:** Custom metric aggregators (arithmetic intensity, FLOP/s, memory BW)

## Existing Infrastructure

### nvbench_integration.h
- Provides `NOVA_BENCHMARK_KERNEL` and `NOVA_BENCHMARK_MEMORY_BANDWIDTH` macros
- Template functions for memory and compute throughput benchmarking
- Depends on NVBlox library

### health_metrics.h
- `HealthMetrics` struct with device utilization, memory, errors, temperature, power
- `HealthMonitor` class with snapshot and export (JSON/CSV)

### nvtx_extensions.h
- `NVTXDomains` with Memory, Device, Algo, API, Production domains
- `ScopedRange` template for RAII-scoped profiling ranges
- Disabled when `NOVA_NVTX_ENABLED=0`

## Implementation Strategy

### NVBloxMetricsCollector Class
```cpp
class NVBloxMetricsCollector {
public:
    NVBloxMetricsCollector();
    ~NVBloxMetricsCollector();
    
    void register_metric(const std::string& name, MetricType type);
    void add_sample(const std::string& name, double value);
    void record_kernel(const KernelMetrics& km);
    
    std::vector<KernelMetrics> get_metrics() const;
    std::string to_json() const;
};
```

### KernelMetrics Struct
```cpp
struct KernelMetrics {
    std::string name;
    uint64_t latency_ns;
    double throughput_gflops;
    float sm_occupancy;
    double arithmetic_intensity;
    double memory_bandwidth_gbs;
    uint64_t timestamp_ns;
};
```

### CMake Integration
- Detect NVBlox via `find_package` or CMake `find_path`
- Provide `NOVA_ENABLE_NVBLOX` option (default ON if found)
- Graceful fallback to CUDA events if NVBlox not available

## Dependencies

- CUDA 20+ for event APIs
- Existing `health_metrics.h` patterns
- Existing NVTX domain structure

## Risks

- NVBlox library may not be installed
- Mitigation: Build optional, fallback to CUDA events
