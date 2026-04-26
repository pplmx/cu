# Phase 40 Plan: FUZZ-04 - Fuzzing Artifact Isolation

## Requirement
**FUZZ-04**: Fuzzing artifacts (corpus, crashes) are isolated in dedicated directories

## Implementation

### 1. Create Directory Structure
```bash
tests/fuzz/
├── corpus/
│   ├── memory_pool/    # Seed corpus for memory pool fuzzing
│   ├── algorithm/      # Seed corpus for algorithm fuzzing
│   └── matmul/         # Seed corpus for matmul fuzzing
├── crashes/            # Crash artifacts (gitignored)
└── baseline/           # Baseline corpus for CI comparison
```

### 2. Create .gitignore
Create `tests/fuzz/crashes/.gitignore`:
```
*
!.gitignore
```

### 3. Add to .gitignore
Add to root `.gitignore`:
```
# Fuzzing crashes
tests/fuzz/crashes/*
!tests/fuzz/crashes/.gitignore
```

### 4. Create Baseline Corpus
Create initial seed files in `tests/fuzz/baseline/`:
- `memory_pool_seed.bin` - minimal memory pool input
- `algorithm_seed.bin` - minimal algorithm input
- `matmul_seed.bin` - minimal matmul input

### 5. Document Structure
Create `tests/fuzz/README.md`:
```markdown
# Fuzz Testing

This directory contains fuzzing infrastructure for Nova.

## Directory Structure

- `corpus/` - Seed corpus for each fuzz target (committed)
- `crashes/` - Crash artifacts (gitignored, for local analysis)
- `baseline/` - Baseline corpus for CI comparison

## Running Fuzz Tests

```bash
# Build with fuzzing
cmake -B build -DNOVA_BUILD_FUZZ_TESTS=ON

# Run memory pool fuzzing
make fuzz_memory_pool

# Run algorithm fuzzing
make fuzz_algorithms

# Run matmul fuzzing
make fuzz_matmul
```

## CI Fuzzing

Fuzz tests run in CI with corpus baseline comparison:
- Minimum corpus size: 1000 entries
- Crash detection fails CI
- Corpus regression fails CI
```

## Verification
1. Verify directory structure exists
2. Verify crashes directory is gitignored
3. Verify README documents usage
