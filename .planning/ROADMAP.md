# Roadmap — v2.6 Transformer & Inference Optimization

## Phases

- [ ] **Phase 69: FlashAttention Integration** - Attention backend selection, IO-aware kernel, stable softmax
- [ ] **Phase 70: Paged KV Cache Foundation** - Block allocator, LRU eviction, prefix caching
- [ ] **Phase 71: Paged Attention Integration** - Block manager, block tables, CPU-GPU sync
- [ ] **Phase 72: Sequence Manager & Scheduler** - Multi-sequence support, continuous batching, GQA/MQA
- [ ] **Phase 73: Sequence Parallelism Extension** - TP/SP integration, ring attention
- [ ] **Phase 74: Integration & Testing** - CUDA Graphs, NVTX, benchmarks

---

## Phase Details

### Phase 69: FlashAttention Integration

**Goal:** FlashAttention-2/3 kernel integration with attention backend selection and stable softmax

**Depends on:** None (foundation)

**Requirements:** FA-01, FA-02, FA-03, FA-04

**Success Criteria** (what must be TRUE):
1. User can select attention backend via `AttentionBackend` enum (Standard/FlashAttention/PagedAttention)
2. FlashAttention forward pass produces output matching standard attention within 1e-3 relative error
3. Stable softmax with max subtraction prevents numerical overflow for large sequence lengths
4. Backward pass computes correct gradients summing to input gradients across batch dimension
5. Workspace allocation is dynamic based on query shape, not fixed at initialization

**Plans:** TBD

**UI hint:** no

---

### Phase 70: Paged KV Cache Foundation

**Goal:** Memory-efficient KV cache allocation with block-based management and LRU eviction

**Depends on:** Phase 69

**Requirements:** KV-01, KV-02, KV-03, KV-04

**Success Criteria** (what must be TRUE):
1. User can allocate KV cache blocks of fixed power-of-2 sizes (16/32/64 tokens) in O(1) from freelist
2. LRU eviction triggers automatically when free_blocks falls below configured threshold
3. Prefix hash lookup returns cached KV blocks for shared conversation prefixes, avoiding recomputation
4. User can query KVCacheStats showing total/used/free blocks and fragmentation percentage
5. Block allocator handles concurrent allocation/deallocation from multiple sequences safely

**Plans:** TBD

**UI hint:** no

---

### Phase 71: Paged Attention Integration

**Goal:** PagedAttention combining FlashAttention with block-based KV cache management

**Depends on:** Phase 70

**Requirements:** PA-01, PA-02, PA-03, PA-04

**Success Criteria** (what must be TRUE):
1. BlockManager.create_sequence returns valid block table mapping logical to physical blocks
2. append_tokens allocates additional physical blocks and updates block table atomically
3. cudaStreamSynchronize called on dedicated sync stream before attention kernel launch
4. Paged attention output matches contiguous attention output within 1e-3 relative error
5. Out-of-bounds block table access returns error rather than reading invalid memory

**Plans:** TBD

**UI hint:** no

---

### Phase 72: Sequence Manager & Scheduler

**Goal:** Multi-sequence orchestration with continuous batching and GQA/MQA support

**Depends on:** Phase 71

**Requirements:** SCHED-01, SCHED-02, SCHED-03

**Success Criteria** (what must be TRUE):
1. Multiple sequences coexist with independent KV cache state and no interference
2. Batched forward pass processes variable-length sequences correctly via iteration-level scheduling
3. GQA/MQA attention produces correct output when num_kv_heads < num_q_heads
4. New sequences can be added to active batch without blocking existing inference
5. Completed sequences release KV cache blocks back to allocator for reuse

**Plans:** TBD

**UI hint:** no

---

### Phase 73: Sequence Parallelism Extension

**Goal:** Distributed attention computation across tensor parallel ranks for long context support

**Depends on:** Phase 72

**Requirements:** SP-01, SP-02, SP-03

**Success Criteria** (what must be TRUE):
1. Sequence attention output across TP ranks matches single-GPU result
2. Ring sequence parallelism handles sequences up to 128K tokens without OOM
3. TP communicator correctly reduces sequence parallel output via all-reduce
4. Ring attention communicates KV projections across ranks with minimal synchronization
5. Sequence parallelism disables gracefully on single-GPU configurations

**Plans:** TBD

**UI hint:** no

---

### Phase 74: Integration & Testing

**Goal:** End-to-end validation with CUDA Graphs, observability, and performance benchmarks

**Depends on:** Phase 73

**Requirements:** All previous requirements

**Success Criteria** (what must be TRUE):
1. CUDA Graph capture/replay works with dynamic block allocation for paged attention
2. NVTX annotations mark inference phases (prefill, decode, attention, scheduling)
3. Throughput benchmark shows >2x speedup vs. standard attention for 1K+ sequence lengths
4. Memory efficiency benchmark demonstrates <4% KV cache waste vs. 60-80% in naive allocation
5. All 18 v2.6 requirements pass integration tests with backward compatibility preserved

**Plans:** TBD

**UI hint:** no

---

## Coverage Summary

| Category | Requirements | Coverage |
|----------|--------------|----------|
| FlashAttention | FA-01, FA-02, FA-03, FA-04 | 100% |
| KV Cache Management | KV-01, KV-02, KV-03, KV-04 | 100% |
| Paged Attention | PA-01, PA-02, PA-03, PA-04 | 100% |
| Scheduler & Batching | SCHED-01, SCHED-02, SCHED-03 | 100% |
| Sequence Parallelism | SP-01, SP-02, SP-03 | 100% |
| **Total** | **18** | **100%** |

---

## Phase Dependencies

```
Phase 69 (FlashAttention)
       │
       ▼
Phase 70 (KV Cache Foundation)
       │
       ▼
Phase 71 (Paged Attention)
       │
       ▼
Phase 72 (Scheduler)
       │
       ▼
Phase 73 (Sequence Parallelism)
       │
       ▼
Phase 74 (Integration & Testing)
```

---

## New Modules

- `cuda/inference/` - Inference orchestration layer
  - `block_manager.h` - Paged attention block manager
  - `sequence_manager.h` - Sequence lifecycle management
  - `scheduler.h` - Batching and scheduling
- `cuda/algo/flash_attention.h` - FlashAttention kernel integration
- `cuda/memory/kv_cache_allocator.h` - Block-based KV cache allocator
- `cuda/distributed/sequence_parallel.h` - Ring attention implementation

---

*Roadmap created: 2026-04-29*
*Milestone: v2.6 Transformer & Inference Optimization*
*Phases: 69-74 (continuation from v2.5 which ended at Phase 68)*
