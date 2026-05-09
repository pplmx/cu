# Phase 81: Extended Formats + Roofline Analysis - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>

## Phase Boundary

Users can store sparse matrices in HYB (Hybrid ELL+COO) format for irregular matrices, classify kernel performance using Roofline model, and export analysis data in JSON format for visualization.

</domain>

<decisions>

## Implementation Decisions

### HYB Format Design

- HYB = ELL + COO hybrid for irregular sparse matrices
- ELL part: stores regular rows (above density threshold)
- COO part: stores irregular rows (below threshold) as coordinate list
- Partition: rows with nnz > threshold → ELL, others → COO
- Threshold: configurable (default: ELL max_nnz_per_row / 2)

### HYB Storage Layout

- ELL portion: max_nnz_ell rows × max_nnz_per_row (similar to ELL)
- COO portion: irregular rows stored as (row, col, value) triples
- Partition metadata: row_to_format[], ELL row count

### HYB SpMV Algorithm

- Iterate through ELL portion: standard ELL SpMV
- Iterate through COO portion: scatter-add pattern
- ELL part: regular, efficient access patterns
- COO part: handles tail of irregular rows

### Roofline Enhancement

- Extend existing RooflineAnalyzer with classification thresholds
- Add visualization-ready data structures
- Support comparison between multiple kernels

### JSON Export

- Structured JSON with kernel metrics, peaks, and classification
- Compatible with standard JSON visualization tools
- Include metadata: device name, precision, timestamp

### the agent's Discretion

- Threshold calculation strategy for ELL/COO partition
- JSON schema design
- Test matrix generation for irregular patterns

</decisions>

<code_context>

## Existing Code Insights

### Reusable Assets from Phase 79

- `SparseMatrixELL<T>` class for ELL storage
- ELL SpMV implementation pattern
- `sparse_mv()` interface

### Reusable Assets from Phase 80

- `RooflineAnalyzer` class
- `RooflineMetrics` struct
- `DevicePeaks` for theoretical peaks

### Integration Points

- New file: `include/cuda/sparse/hyb_matrix.hpp`
- Extend: `include/cuda/sparse/roofline.hpp` (JSON export)
- Tests: `tests/sparse/hyb_roofline_test.cpp`

</code_context>

<specifics>

## Specific Ideas

No specific requirements — follow standard approaches:

- HYB format: standard ELL+COO hybrid (Bell et al.)
- Roofline classification: compare AI * bandwidth vs peak compute
- JSON: standard structure with nested objects

</specifics>

<deferred>

## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
