# Requirements: Nova CUDA Library

**Last Updated:** 2026-04-24
**Current Milestone:** v1.2 Complete
**Next Milestone:** v1.3 Planning Needed

## Milestone Summary

| Milestone | Status | Requirements | Notes |
|-----------|--------|--------------|-------|
| v1.0 Production Release | ✅ Shipped | 58 (PERF, BMCH, ASYNC, POOL, FFT, RAY, GRAPH, NN) | 2026-04-24 |
| v1.1 Multi-GPU Support | ✅ Shipped | 13 (MGPU-01 to MGPU-13) | 2026-04-24 |
| v1.2 Toolchain Upgrade | ✅ Shipped | 9 (TC-01 to TC-09) + 2 (TC-10, TC-11) | 2026-04-24 |

See archived requirements in `.planning/milestones/` for full details.

---

## v1.3 Candidates (Planning Needed)

### NCCL Integration

- [ ] **NCCL-01**: NCCL library detection and linking
- [ ] **NCCL-02**: NCCL-based all-reduce (replaces P2P ring algorithm)
- [ ] **NCCL-03**: NCCL-based broadcast and all-gather
- [ ] **NCCL-04**: Multi-node NCCL communication (future)

### Tensor Parallelism

- [ ] **TENS-01**: Column-parallel matmul with all-reduce
- [ ] **TENS-02**: Row-parallel matmul with all-gather
- [ ] **TENS-03**: Tensor parallelism utilities

### Pipeline Parallelism

- [ ] **PIPE-01**: Pipeline stage abstraction
- [ ] **PIPE-02**: 1F1B (one-forward-one-backward) scheduling
- [ ] **PIPE-03**: Micro-batch management

### Additional

- [ ] **DIST-01**: Distributed batch normalization
- [ ] **TOPO-01**: NVLink-aware topology scheduling

---

## Out of Scope

- Multi-node computation (multiple hosts, not just multiple GPUs)
- Python bindings — separate project
- CUDA MPS multi-process management — library feature, not deployment config
- NVSHMEM — requires InfiniBand hardware, out of scope for single-node

---

## Current Tech Stack

- **C++ Standard:** C++23
- **CUDA Standard:** CUDA 20
- **CMake Version:** 4.0+
- **Test Framework:** GoogleTest v1.17.0
- **Test Count:** 444 tests passing

---

Run `/gsd-new-milestone` to start v1.3 planning.

---

*Requirements updated: 2026-04-24 after v1.2 milestone completion*
