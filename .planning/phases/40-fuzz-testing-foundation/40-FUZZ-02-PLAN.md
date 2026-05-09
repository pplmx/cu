# Phase 40 Plan: FUZZ-02 - Algorithm Fuzzing

## Requirement

**FUZZ-02**: User can run property-based fuzzing on algorithm inputs (reduce, scan, sort)

## Implementation

### 1. Create Algorithm Fuzz Target

Create `tests/fuzz/algorithm_fuzz.h`:

```cpp
#include <nova/algo/reduce.hpp>
#include <nova/algo/scan.hpp>
#include <nova/algo/sort.hpp>
#include <fuzzer/FuzzedDataProvider.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    FuzzedDataProvider provider(data, size);

    // Generate operation type (0=reduce, 1=scan, 2=sort)
    auto op = provider.ConsumeIntegralInRange<int>(0, 2);
    auto count = provider.ConsumeIntegralInRange<size_t>(1, 16384);

    // Generate test data
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = provider.ConsumeFloatingPoint<float>();
    }

    switch (op) {
        case 0: { // reduce
            nova::reduce(input.data(), count, 0.0f, [](float a, float b) { return a + b; });
            break;
        }
        case 1: { // scan
            std::vector<float> output(count);
            nova::inclusive_scan(input.data(), output.data(), count);
            break;
        }
        case 2: { // sort
            std::vector<float> sorted = input;
            nova::sort(sorted.data(), sorted.size());
            break;
        }
    }

    return 0;
}
```

### 2. Create CMake Target

```cmake
add_fuzz_target(algorithm_fuzz algorithm_fuzz.cpp)
target_link_libraries(algorithm_fuzz PRIVATE nova ${FUZZER_LIB})
```

### 3. Add Make Target

```cmake
add_custom_target(fuzz_algorithms
    COMMAND $<TARGET_FILE:algorithm_fuzz> tests/fuzz/corpus/algorithm -max_total_time=60
    COMMENT "Running algorithm fuzzing tests"
)
```

## Verification

1. Run `make fuzz_algorithms`
2. Verify crash count (should be 0)
3. Verify edge cases discovered in corpus
