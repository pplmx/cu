# Fuzz Testing

This directory contains fuzzing infrastructure for Nova using libFuzzer.

## Directory Structure

- `corpus/` - Seed corpus for each fuzz target (committed to git)
  - `memory_pool/` - Seeds for memory pool fuzzing
  - `algorithm/` - Seeds for algorithm fuzzing
  - `matmul/` - Seeds for matmul fuzzing
- `crashes/` - Crash artifacts (gitignored, for local analysis)
- `baseline/` - Baseline corpus for CI comparison

## Running Fuzz Tests

### Build with Fuzzing

```bash
cmake -B build -DNOVA_BUILD_FUZZ_TESTS=ON
cmake --build build
```

### Run Individual Fuzz Targets

```bash
# Memory pool fuzzing
./build/bin/memory_pool_fuzz tests/fuzz/corpus/memory_pool -max_total_time=60

# Algorithm fuzzing
./build/bin/algorithm_fuzz tests/fuzz/corpus/algorithm -max_total_time=60

# Matmul fuzzing
./build/bin/matmul_fuzz tests/fuzz/corpus/matmul -max_total_time=60
```

## CMake Targets

When `NOVA_BUILD_FUZZ_TESTS=ON`:

- `make fuzz_memory_pool` - Run memory pool fuzzing
- `make fuzz_algorithms` - Run algorithm fuzzing
- `make fuzz_matmul` - Run matmul fuzzing
- `make fuzz_all` - Run all fuzzing targets

## CI Fuzzing

Fuzz tests run in CI with corpus baseline comparison:

- Minimum corpus size: 1000 entries
- Crash detection fails CI
- Corpus regression fails CI

## Fuzzer Details

### Memory Pool Fuzzing (FUZZ-01)
Tests allocation/deallocation patterns and edge cases in memory management.

### Algorithm Fuzzing (FUZZ-02)
Tests reduce, scan, and sort operations with varied input sizes and values.

### Matmul Fuzzing (FUZZ-03)
Tests matrix multiplication with varied tensor shapes and precision modes.
