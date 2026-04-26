# Nova Coverage Infrastructure

This directory contains scripts and configuration for generating code coverage reports.

## Quick Start

```bash
# Build with coverage
cmake -B build-coverage -DNOVA_COVERAGE=ON
cmake --build build-coverage

# Run tests
ctest --test-dir build-coverage

# Generate coverage report
./scripts/generate_coverage.sh build-coverage
# Output: build-coverage/coverage/index.html
```

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_coverage.sh` | Generate full coverage report |
| `coverage_gaps.sh` | Show untested functions |
| `coverage_summary.sh` | Per-module breakdown |

## CMake Options

| Option | Description |
|--------|-------------|
| `NOVA_COVERAGE` | Enable coverage instrumentation |
| `NOVA_COVERAGE_MIN` | Minimum required coverage (default: 80) |

## Coverage Report Structure

```
build/coverage/
├── index.html          # Main HTML report
├── lcov/               # Raw lcov data files
│   ├── app.info
│   └── total.info
├── module/             # Per-module reports
│   ├── memory.info
│   ├── algo.info
│   └── neural.info
└── gaps/               # Coverage gaps
    └── untested_functions.txt
```

## Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| COVR-01 | HTML coverage report | ✅ |
| COVR-02 | Coverage gap analysis | ✅ |
| COVR-03 | Per-module breakdown | ✅ |

## Dependencies

- lcov (Linux: `apt install lcov`)
- genhtml (part of lcov)
- Clang or GCC with coverage support

## CI Integration

See Phase 43 for CI coverage gates.
