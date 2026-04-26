# Phase 40 Plan: FUZZ-03 - Matmul Fuzzing

## Requirement
**FUZZ-03**: User can run property-based fuzzing on matmul with varied tensor shapes

## Implementation

### 1. Create Matmul Fuzz Target
Create `tests/fuzz/matmul_fuzz.h`:
```cpp
#include <nova/neural/matmul.hpp>
#include <fuzzer/FuzzedDataProvider.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    FuzzedDataProvider provider(data, size);
    
    // Generate matrix dimensions
    auto M = provider.ConsumeIntegralInRange<size_t>(1, 1024);
    auto K = provider.ConsumeIntegralInRange<size_t>(1, 1024);
    auto N = provider.ConsumeIntegralInRange<size_t>(1, 1024);
    
    // Generate precision mode
    auto precision = provider.ConsumeIntegralInRange<int>(0, 2);
    
    // Create input matrices
    nova::Buffer<float> A(M * K);
    nova::Buffer<float> B(K * N);
    nova::Buffer<float> C(M * N);
    
    // Fill with fuzzed data
    for (size_t i = 0; i < M * K; ++i) A[i] = provider.ConsumeFloatingPoint<float>();
    for (size_t i = 0; i < K * N; ++i) B[i] = provider.ConsumeFloatingPoint<float>();
    
    // Execute matmul
    nova::MatmulParams params;
    params.precision = static_cast<nova::Precision>(precision);
    
    nova::matmul(A.data(), B.data(), C.data(), M, K, N, params);
    
    return 0;
}
```

### 2. Create CMake Target
```cmake
add_fuzz_target(matmul_fuzz matmul_fuzz.cpp)
target_link_libraries(matmul_fuzz PRIVATE nova ${FUZZER_LIB})
```

### 3. Add Make Target
```cmake
add_custom_target(fuzz_matmul
    COMMAND $<TARGET_FILE:matmul_fuzz> tests/fuzz/corpus/matmul -max_total_time=60
    COMMENT "Running matmul fuzzing tests"
)
```

## Verification
1. Run `make fuzz_matmul`
2. Verify edge case discoveries logged
3. Verify numerical stability maintained
