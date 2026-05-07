# Phase 103: KV Cache Foundation - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can manage KV cache with streaming, prefix sharing, and memory efficiency features for transformer inference. This phase extends the existing KVCacheAllocator with async streaming, improved prefix caching, attention sink handling, fragmentation monitoring, L2 persistence, and persistent KV support for CUDA Graphs.
</domain>

<decisions>
## Implementation Decisions

### Architecture Pattern
- Extend existing `KVCacheAllocator` class rather than creating new allocator
- Add StreamingCacheManager as a wrapper/decorator around KVCacheAllocator
- Use RAII patterns consistent with existing codebase

### Prefix Caching
- Extend existing prefix_hash mechanism for cross-sequence sharing
- Use block-level hash with sequence ID binding
- Enable reference-counted sharing (fork-on-diverge pattern)

### Attention Sink Handling
- Separate sink blocks from LRU eviction pool
- Store attention sink blocks with infinite "recency" score
- Configurable number of sink positions (default: 4-8)

### L2 Persistence
- Use CUDA managed memory hints for L2 cache persistence
- Provide configuration API for persistence scopes
- Support both streaming (no persistence) and iterative (persistence) modes

### CUDA Graph Support
- Design for stateless attention operations that can be captured
- Avoid capturing mutable state in CUDA Graphs
- Use conditional graph nodes for variable sequence lengths

### Agent's Discretion
- Block size selection (16/32/64) - use existing config or add runtime detection
- Fragmentation compaction trigger threshold - default 30% as per research
- Eviction policy selection - LRU is default, consider importance-weighted option

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cuda/memory/KVCacheAllocator` - existing allocator with block management, LRU, prefix hash
- `cuda/memory/KVCacheBlock` - existing block structure with sequence_id, block_id
- `cuda/inference/BlockManager` - wraps KVCacheAllocator, manages sequences
- `cuda/stream/Stream` - existing stream abstraction for async operations

### Established Patterns
- RAII with unique_ptr for resource management
- std::shared_mutex for thread-safe access
- Config structs with sensible defaults
- Buffer<T> for GPU memory management
- Stats structs for monitoring

### Integration Points
- KVCacheAllocator::allocate/append/free for block lifecycle
- KVCacheAllocator::get_stats() for fragmentation monitoring
- BlockManager for sequence management
- Stream for async operations

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Reference vLLM's prefix caching design and FlashInfer page API for integration patterns.
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>
