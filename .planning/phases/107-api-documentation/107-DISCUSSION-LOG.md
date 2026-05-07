# Phase 107: API Documentation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 107-api-documentation
**Areas discussed:** Documentation scope and priority, Doxygen group organization, Example code strategy, Performance notes coverage

---

## Documentation scope and priority

| Option | Description | Selected |
|--------|-------------|----------|
| All public headers | Comprehensive coverage. Every .hpp in include/cuda/ gets documented. | ✓ |
| Top-level API layer only | Focus on include/cuda/api/*.hpp — the user-facing API surface. | |
| By layer priority | Document in order: memory → device → algo → api → other | |

**User's choice:** All public headers
**Notes:** Comprehensive coverage across all public headers prioritized.

---

## Doxygen group organization

| Option | Description | Selected |
|--------|-------------|----------|
| Five-layer architecture | Match existing code structure: memory, device, algo, api, observability | |
| By capability | Group by feature: compute, memory, distributed, inference, production | |
| Keep existing + extend | Use Phase 37 groups (memory, device, algo, api) and add new ones as needed | ✓ |

**User's choice:** Keep existing + extend (agent recommendation accepted)
**Notes:** Agent recommended keeping five-layer architecture groups and extending as needed, noting consistency with Phase 37 work and natural mapping to code structure.

---

## Example code strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Key public APIs only | Examples for high-level public API functions in cuda/api/ | |
| Per-layer entry points | One example per module showing basic usage (memory, device, algo, etc.) | ✓ |
| All exported functions | Examples for every public (non-internal) function with non-trivial parameters | |

**User's choice:** Per-layer entry points (agent recommendation accepted)
**Notes:** Agent recommended per-layer entry points to avoid cluttering simple APIs. User agreed that existing `examples/` directory already provides comprehensive runnable examples.

---

## Performance notes coverage

| Option | Description | Selected |
|--------|-------------|----------|
| Kernel launchers only | Only add @note to explicit kernel launch functions | |
| Compute-intensive APIs | Add to algorithms, matrix ops, attention mechanisms, solvers | ✓ |
| All public functions | Add performance notes to every function that has non-trivial cost | |

**User's choice:** Compute-intensive APIs (agent recommendation accepted)
**Notes:** Agent recommended covering algo, linalg, neural, sparse solvers, NCCL, and inference modules. Agreed to omit simple getters/setters and RAII wrappers.

---

## Agent's Discretion

- Specific @defgroup names for new modules (sparse, quantize, gnn) — agent to decide based on Phase 37 naming convention
- Detailed performance note content format — agent to follow best practices
- Cross-reference granularity — which related functions to link via @see

## Deferred Ideas

None
