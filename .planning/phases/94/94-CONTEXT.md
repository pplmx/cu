# Phase 94: Kernel Fusion Analysis — Context

**Created:** 2026-05-02
**Milestone:** v2.11 Performance Tooling

## Requirements

- **FUSION-01:** KernelFusionAnalyzer with op-to-op fusion opportunity detection
- **FUSION-02:** Fusion profitability model (break-even analysis based on launch overhead)
- **FUSION-03:** Fusion recommendation engine with confidence levels

## Prior Work

### Phase 93: NVBlox Foundation
- `KernelMetrics` struct with latency, throughput, occupancy, AI, BW
- `NVBloxMetricsCollector` for metric collection
- `KernelProfiler` for CUDA event-based profiling
- `OccupancyCalculator` for theoretical occupancy

### Existing Infrastructure

#### graph_executor.h (v2.4)
- `GraphExecutor` for CUDA graph capture
- `MemoryNode`, `AlgoWrapper` for graph construction

#### fused_matmul_bias_act.cu (v2.2)
- Existing fused kernel: matmul + bias + activation

## Implementation Strategy

### KernelFusionAnalyzer Class
```cpp
class KernelFusionAnalyzer {
public:
    void analyze_operation_graph(const OpGraph& graph);
    std::vector<FusionOpportunity> detect_opportunities();
    FusionRecommendation generate_recommendation(const FusionOpportunity& opp);
    
private:
    std::vector<FusionPattern> patterns_;
    FusionProfitabilityModel model_;
};
```

### FusionPattern Library
Common fusion patterns to detect:
- matmul + bias + activation (ReLU, GeLU, SiLU)
- conv + bias + activation
- relu + pooling
- element-wise ops (multiple into one)
- reduction + normalization

### FusionProfitabilityModel
```cpp
struct FusionProfitabilityModel {
    double launch_overhead_us;  // Typical kernel launch overhead
    double memory_coalescing_benefit;
    double register_pressure_cost;
    
    bool is_profitable(const FusionOpportunity& opp) const;
    double profitability_score(const FusionOpportunity& opp) const;
};
```

### Confidence Levels
- **HIGH**: Exact pattern match (e.g., matmul+bias+ReLU sequence)
- **MEDIUM**: Heuristic match (memory bound ops adjacent)
- **LOW**: Statistical match (historical performance improvement)

## Dependencies

- Phase 93: `KernelMetrics`, `NVBloxMetricsCollector`
- Existing: `GraphExecutor`, CUDA graph capture patterns

## Risks

- Fusion heuristics may not accurately predict performance
- Mitigation: Provide confidence levels and allow manual override
