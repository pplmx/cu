# Roadmap — v2.2 Comprehensive Enhancement

## Milestone Summary

| Metric | Value |
|--------|-------|
| Milestone | v2.2 Comprehensive Enhancement |
| Requirements | 18 |
| Phases | 6 |
| Started | 2026-04-27 |

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 48 | Kernel Fusion & Autotuning | Fused kernels and hardware-aware tuning | PERF-01, PERF-03 | 4 criteria |
| 49 | Memory Optimization | Enhanced memory management and compression | PERF-02, PERF-04 | 4 criteria |
| 50 | Transformer & Loss | Transformer components and loss functions | OP-01, OP-02, OP-03, OP-04, OP-05 | 5 criteria |
| 51 | Optimizers | AdamW, LAMB, gradient clipping | OP-06, OP-07, OP-08 | 3 criteria |
| 52 | Tooling | Debug, profiling, developer utilities | TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05, TOOL-06 | 6 criteria |
| 53 | Documentation | User guides, architecture, API docs | DOC-01, DOC-02, DOC-03, DOC-04 | 4 criteria |

---

## Phase 48: Kernel Fusion & Autotuning

**Goal:** Enable kernel fusion for chained operations and autotuning infrastructure

**Requirements:** PERF-01, PERF-03

**Success Criteria:**
1. User can fuse matmul + bias + activation into single kernel
2. User can configure fusion patterns via policy
3. User can run autotuning for block sizes on target GPU
4. Autotuned parameters persist in config file

**Dependencies:** None (foundation phase)

**Key Decisions:**
- Use CUDA fusion API where available
- Provide fallback manual fusion for older CUDA versions
- Autotuning uses grid search with warmup runs

---

## Phase 49: Memory Optimization

**Goal:** Enhanced memory management with pool tuning and compression

**Requirements:** PERF-02, PERF-04

**Success Criteria:**
1. User can enable adaptive memory pool sizing
2. User can configure memory pool based on workload profile
3. User can enable checkpoint compression with ZSTD
4. Memory compression shows >50% size reduction for typical checkpoints

**Dependencies:** None (uses existing memory infrastructure)

**Key Decisions:**
- Workload profiling uses histogram-based learning
- Compression uses ZSTD with level 3 default
- Pool sizing respects existing device limits

---

## Phase 50: Transformer & Loss

**Goal:** Transformer components and loss functions

**Requirements:** OP-01, OP-02, OP-03, OP-04, OP-05

**Success Criteria:**
1. User can run multi-head attention with configurable heads
2. User can apply sinusoidal or learned positional encoding
3. User can compute numerically stable cross-entropy loss
4. User can compute focal loss with configurable gamma
5. User can compute contrastive loss with temperature

**Dependencies:** Phase 44 (uses existing matmul infrastructure)

**Key Decisions:**
- Attention supports both self-attention and cross-attention
- Positional encoding is swappable via policy
- Loss functions use log-sum-exp for numerical stability

---

## Phase 51: Optimizers

**Goal:** Implement AdamW and LAMB optimizers with gradient clipping

**Requirements:** OP-06, OP-07, OP-08

**Success Criteria:**
1. User can instantiate AdamW with configurable lr/weight_decay
2. User can instantiate LAMB with layer-wise LR decay
3. User can apply gradient clipping with configurable norm threshold

**Dependencies:** Phase 50 (uses transformer components for testing)

**Key Decisions:**
- LAMB uses trust ratio clipping per layer
- Gradient clipping uses global norm by default
- All optimizers support mixed precision (FP16/BF16)

---

## Phase 52: Tooling

**Goal:** Debugging, profiling enhancements, and developer utilities

**Requirements:** TOOL-01, TOOL-02, TOOL-03, TOOL-04, TOOL-05, TOOL-06

**Success Criteria:**
1. User can run memory sanitizer with ASAN/UBSAN integration
2. User can detect bank conflict patterns in shared memory access
3. User can visualize kernel timeline in Chrome trace format
4. User can measure memory bandwidth utilization per kernel
5. User can generate kernel boilerplate via nova-codegen CLI
6. User can run automated benchmark comparison against baseline

**Dependencies:** Phases 48-49 (uses performance infrastructure)

**Key Decisions:**
- Sanitizer integration via CMake options
- Timeline export uses Chrome trace format (standard)
- CLI code generation uses Jinja2 templates

---

## Phase 53: Documentation

**Goal:** Comprehensive documentation covering architecture and usage

**Requirements:** DOC-01, DOC-02, DOC-03, DOC-04

**Success Criteria:**
1. User can follow transformer implementation tutorial
2. User can understand five-layer architecture from docs
3. User can read decision rationale for key design choices
4. User can find code examples for all major APIs

**Dependencies:** All previous phases (documents what was built)

**Key Decisions:**
- Tutorial is narrative-style with runnable examples
- Architecture docs use mermaid diagrams
- Decision rationale stored in ARCHITECTURE.md

---

## Requirement Coverage Matrix

| REQ-ID | Phase | Covered |
|--------|-------|---------|
| PERF-01 | 48 | ✅ |
| PERF-02 | 49 | ✅ |
| PERF-03 | 48 | ✅ |
| PERF-04 | 49 | ✅ |
| OP-01 | 50 | ✅ |
| OP-02 | 50 | ✅ |
| OP-03 | 50 | ✅ |
| OP-04 | 50 | ✅ |
| OP-05 | 50 | ✅ |
| OP-06 | 51 | ✅ |
| OP-07 | 51 | ✅ |
| OP-08 | 51 | ✅ |
| TOOL-01 | 52 | ✅ |
| TOOL-02 | 52 | ✅ |
| TOOL-03 | 52 | ✅ |
| TOOL-04 | 52 | ✅ |
| TOOL-05 | 52 | ✅ |
| TOOL-06 | 52 | ✅ |
| DOC-01 | 53 | ✅ |
| DOC-02 | 53 | ✅ |
| DOC-03 | 53 | ✅ |
| DOC-04 | 53 | ✅ |

**All 18 requirements mapped across 6 phases.**

---

## Execution Order

```
Phase 48 (Kernel Fusion & Autotuning) ← Foundation
         ↓
Phase 49 (Memory Optimization) ← Independent, can parallel
         ↓
Phase 50 (Transformer & Loss) ← Builds on fusion
         ↓
Phase 51 (Optimizers) ← Builds on transformer
         ↓
Phase 52 (Tooling) ← Independent, can parallel
         ↓
Phase 53 (Documentation) ← Requires all previous phases
```

---

*Roadmap created: 2026-04-27 for v2.2 Comprehensive Enhancement*
*6 phases, 18 requirements, all covered*
