# Phase 37: API Reference

**Milestone:** v1.9 Documentation
**Status:** Planning

## Goal

Generate comprehensive API documentation from source code

## Requirements

| ID | Description |
|----|-------------|
| API-01 | Doxygen configuration generates HTML documentation |
| API-02 | All public headers have documented function signatures |
| API-03 | Grouped documentation by module |
| API-04 | Cross-references link related functions and types |

## Success Criteria

1. Developer can run `doxygen` and generate HTML documentation
2. All public functions in headers have Doxygen comments
3. Documentation is grouped by module (memory, device, algo, api)
4. Cross-references link related functions and types

## Implementation Plan

### 1. Create Doxygen Configuration

```bash
doxygen -g Doxyfile
```

Key settings:

- OUTPUT_DIRECTORY = docs/api
- GENERATE_HTML = YES
- INPUT = include/
- FILE_PATTERNS = *.hpp*.h
- ENABLE_PREPROCESSING = YES
- EXPAND_PREPROC = YES

### 2. Add Doxygen Comments to Headers

Group by module:

- @defgroup memory Memory Management
- @defgroup device Device Management
- @defgroup algo Algorithms
- @defgroup api Public API

### 3. Module Structure

```text
docs/api/html/
├── index.html
├── group__memory.html
├── group__device.html
├── group__algo.html
└── group__api.html
```

---

## Context created: 2026-04-26
