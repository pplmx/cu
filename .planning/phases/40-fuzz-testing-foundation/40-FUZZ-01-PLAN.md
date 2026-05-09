# Phase 40 Plan: FUZZ-01 - Memory Pool Fuzzing

## Requirement

**FUZZ-01**: User can run property-based fuzzing on memory pool operations

## Implementation

### 1. Create Fuzz Target Header

Create `tests/fuzz/memory_pool_fuzz.h`:

```cpp
#include <nova/memory/memory_pool.hpp>
#include <fuzzer/FuzzedDataProvider.h>

// Fuzz target for memory pool operations
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    FuzzedDataProvider provider(data, size);

    // Generate allocation pattern
    auto alloc_size = provider.ConsumeIntegralInRange<size_t>(1, 1024 * 1024);
    auto alignment = provider.ConsumeIntegralInRange<size_t>(1, 256);

    // Create temporary pool for fuzzing
    nova::MemoryPool pool;

    // Allocate and free in various patterns
    auto* ptr = pool.allocate(alloc_size, alignment);
    if (ptr) {
        pool.deallocate(ptr, alloc_size, alignment);
    }

    return 0;
}
```

### 2. Create CMake Target

Add to `tests/CMakeLists.txt`:

```cmake
# Fuzz Testing
if(NOT NOVA_BUILD_FUZZ_TESTS)
    find_library(FUZZER_LIB Fuzzer REQUIRED)

    add_fuzz_target(memory_pool_fuzz memory_pool_fuzz.cpp)
    target_link_libraries(memory_pool_fuzz PRIVATE nova ${FUZZER_LIB})
endif()
```

### 3. Create Corpus Generator

Create `tests/fuzz/corpus/memory_pool/` with sample inputs.

### 4. Add Make Target

Add to root CMakeLists.txt:

```cmake
add_custom_target(fuzz_memory_pool
    COMMAND $<TARGET_FILE:memory_pool_fuzz> tests/fuzz/corpus/memory_pool -max_total_time=60
    COMMENT "Running memory pool fuzzing tests"
)
```

## Verification

1. Run `make fuzz_memory_pool`
2. Verify corpus count displayed
3. Verify no crashes detected
