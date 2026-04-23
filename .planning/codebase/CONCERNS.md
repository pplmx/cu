# Technical Concerns

**Mapped:** 2026-04-23

## Technical Debt

### Limited Error Recovery

- Exceptions propagate but no graceful degradation
- Memory allocations fail completely rather than falling back
- No timeout mechanisms for long-running operations

### Missing Features

- **No async/await patterns** - Operations block until complete
- **No streaming operations** - Must load all data into memory
- **No multi-GPU support** - Single GPU only
- **No memory metrics** - No way to query memory usage

## Known Issues

### Memory Pool

- No defragmentation strategy
- Fixed block sizes may waste memory
- No way to return memory to pool

### Algorithm Limitations

- No size validation on all algorithms
- Empty input handling varies between algorithms
- No performance guarantees for edge cases

## Fragile Areas

### Kernel Launch Configuration

```cpp
// Magic numbers for block/grid size
constexpr int BLOCK_SIZE = 256;
dim3 block(BLOCK_SIZE);
dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
```

These values are hardcoded rather than computed based on device capabilities.

### Template Instantiation

CUDA templates may cause cryptic errors if not instantiated properly. All kernels should be instantiated in implementation files.

### Error Message Quality

Current error messages include file/line but could include:
- Operation name
- Input dimensions
- Device information

## Performance Considerations

### Memory Access Patterns

- No explicit memory coalescing guarantees
- Random access patterns may be inefficient
- No pinned memory for host-device transfers

### Kernel Efficiency

- No shared memory usage in most kernels
- No algorithm-specific tuning (e.g., sort block size)

## Security Considerations

- **No input validation** on buffer sizes
- **No thread safety** for shared state
- **No sanitizers** in default build

## Testing Gaps

- **No performance benchmarks** (only correctness tests)
- **No property-based tests**
- **No fuzzing**
- **No integration tests** for multi-module workflows

## Build Complexity

- CMake configuration can be slow on first run
- FetchContent downloads Google Test at configure time
- CUDA separable compilation increases compile time
