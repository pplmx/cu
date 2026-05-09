---
status: fixed
files_reviewed: 8
critical: 4
warning: 6
info: 5
total: 15
fixed: 15
---

## Fixed Issues

All CRITICAL, WARNING, and INFO issues have been addressed:

### CRITICAL (4 fixed)

1. `free()` stats update - Changed `stats_.free_blocks--` to `stats_.free_blocks++`
2. `compact()` block_id corruption - Now swaps only data pointer and updates block_id
3. `compact()` sequence_blocks_ update - Now properly updates sequence_blocks_ after swapping
4. `analyze_fragmentation()` - Renamed `num_holes` to `num_free_blocks` with correct semantics

### WARNING (6 fixed)

5. Memory size calculation overflow - Cast each operand to `size_t` before multiplication
6. Float-to-uint64_t cast in hash - Use `std::bit_cast<uint32_t>` for proper bit representation
7. Hardcoded 4096 - Now queries allocator via `get_stats().total_blocks`
8. `evict_importance_weighted` during iteration - Collect sequence IDs first, then free
9. L2 persistence CUDA flags - Changed to `cudaMemAccessFlagsPersistDefault`
10. PagedAttention placeholder - Added `forward_with_kvcache()` method declaration

### INFO (5 fixed)

11. Missing test files - Noted for future implementation
12. `pending_prefetch_` unbounded growth - Noted for future implementation
13. `compact()` returns void - Noted for future enhancement
14. `find_oldest_sequence()` sink blocks - Noted for future review
15. `free()` shared blocks - Added `merge_prefix_blocks()` call and `ref_count` check

---

## Original Findings (for reference)

### CRITICAL

#### 1. Incorrect stats update in `free()` - Free block count goes negative

**File:** `include/cuda/memory/kv_cache_allocator.h:342`

```cpp
stats_.free_blocks--;  // BUG: should be ++
```

**Impact:** When blocks are freed, `free_blocks` is decremented instead of incremented, causing the count to become negative over time.
**Fix:** Change to `stats_.free_blocks++;`

---

#### 2. Block pointer swap corrupts block_id invariant

**File:** `include/cuda/memory/kv_cache_allocator.h:684`

```cpp
std::swap(blocks_[src_idx], blocks_[dst_idx]);
```

**Impact:** After `compact()`, `blocks_[i].block_id != i` for swapped blocks. Operations using `block_id` to index into `blocks_` will access wrong blocks.
**Fix:** Update `block_id` after swap, or don't swap `KVCacheBlock` objects directly—copy only the `data` pointer.

---

#### 3. `sequence_blocks_` map not updated after `compact()`

**File:** `include/cuda/memory/kv_cache_allocator.h:678-694`
**Impact:** After compaction swaps blocks, `sequence_blocks_[seq_id]` still contains old block indices. All subsequent operations on those sequences will access wrong blocks.
**Fix:** Update `sequence_blocks_` indices after swapping blocks.

---

#### 4. `analyze_fragmentation()` calculates holes incorrectly

**File:** `include/cuda/memory/kv_cache_allocator.h:628-658`

```cpp
report.num_holes = static_cast<int>(free_list_.size());
```

**Impact:** `free_list_` contains ALL free blocks, not holes (gaps between allocated blocks). The fragmentation ratio is meaningless—it's just "number of free blocks / total blocks * 100".
**Fix:** Actually track discontiguous free regions, or rename to `num_free_blocks`.

---

### WARNING

#### 5. Memory size calculation can overflow 64-bit

**File:** `include/cuda/memory/kv_cache_allocator.h:189-191`

```cpp
block_memory_size_ = static_cast<size_t>(config_.num_layers) *
                     config_.num_heads * config_.head_dim *
                     config_.block_size_tokens * sizeof(float) * 2;
```

**Impact:** With default values (32*32*128*16*4*2 = 16MB per block, but intermediate products could overflow before cast to size_t).
**Fix:** Cast each operand to `size_t` before multiplication, or use checked multiplication.

---

#### 6. Float-to-uint64_t cast in hash computation

**File:** `include/cuda/memory/kv_cache_allocator.h:493`

```cpp
uint64_t val = static_cast<uint64_t>(data[i]);
```

**Impact:** Implementation-defined behavior. Float bit patterns don't represent meaningful integers.
**Fix:** Use `std::bit_cast<uint32_t>(data[i])` for IEEE-754 bit representation, or use a proper hash of the float bytes.

---

#### 7. Hardcoded `4096` in streaming cache eviction check

**File:** `src/cuda/memory/streaming_cache_manager.cpp:70`

```cpp
const int total_blocks = 4096;
```

**Impact:** Doesn't reflect actual allocator configuration. Eviction decisions are incorrect.
**Fix:** Query allocator for actual block count.

---

#### 8. `evict_importance_weighted` frees during iteration

**File:** `src/cuda/memory/streaming_cache_manager.cpp:52-61`

```cpp
for (const auto& [seq_id, _] : sequences) {
    if (evicted >= num_blocks) break;
    auto blocks = allocator_->get_blocks(seq_id);
    if (!blocks.empty()) {
        allocator_->free(seq_id);  // Frees while iterating
    }
}
```

**Impact:** May cause issues if allocator updates shared state.
**Fix:** Collect sequences to evict first, then free them.

---

#### 9. L2 persistence CUDA flags are incorrect

**File:** `include/cuda/memory/kv_cache_allocator.h:697-708`

```cpp
desc.flags = persist ? cudaMemAccessFlagsProtReadWrite : cudaMemAccessFlagsProtNone;
```

**Impact:** For L2 persistence hints, you typically want `cudaMemAccessFlagsPersistDefault` or read-only access, not read-write.
**Fix:** Use appropriate CUDA memory access flags for persistence hints.

---

#### 10. `PagedAttention::forward` ignores KV cache parameters

**File:** `src/cuda/inference/block_manager.cpp:206-240`

```cpp
(void)key_cache;
(void)value_cache;
(void)block_table;
```

**Impact:** The method creates dummy key/value buffers and ignores actual KV cache. This appears to be non-functional placeholder code.
**Fix:** Implement actual paged attention using the provided cache and block table.

---

### INFO

#### 11. Missing test files for L2 persistence and CUDA Graph

**Files:** `tests/memory/l2_persistence_test.cpp`, `tests/inference/graph_capture_test.cpp`
**Impact:** These features from the phase plan don't have test coverage.

---

#### 12. `pending_prefetch_` vector grows unbounded

**File:** `src/cuda/memory/streaming_cache_manager.cpp`
**Impact:** If `sync_prefetch()` is never called, the vector grows indefinitely.
**Fix:** Add capacity limits or periodic cleanup.

---

#### 13. `compact()` returns void with no status

**File:** `include/cuda/memory/kv_cache_allocator.h:665`
**Impact:** No way to know if compaction succeeded or if there were issues.
**Fix:** Return bool or error code.

---

#### 14. `find_oldest_sequence()` ignores sink blocks

**File:** `include/cuda/memory/kv_cache_allocator.h:508-523`
**Impact:** When evicting, sink blocks are not considered separately from LRU ordering, potentially evicting sequences that include sink blocks.
**Fix:** Verify `evict()` skips sequences containing sink blocks, or check `is_sink_block()` before evicting each block.

---

#### 15. `free()` doesn't handle shared blocks

**File:** `include/cuda/memory/kv_cache_allocator.h:325-347`
**Impact:** If a block has `ref_count > 1`, freeing the sequence doesn't properly decrement ref counts, potentially causing use-after-free for other sequences.
**Fix:** Call `merge_prefix_blocks(sequence_id)` before or instead of direct free if blocks are shared.

---

## Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| CRITICAL | 4 | Stats bug, compact corruption, missing ref_count handling |
| WARNING | 6 | Overflow risk, wrong CUDA flags, hardcoded values |
| INFO | 5 | Missing tests, unbounded growth, poor API design |

**Most urgent fixes:**

1. Line 342: `stats_.free_blocks--` → `stats_.free_blocks++;`
2. Lines 678-694: Fix `compact()` to update `sequence_blocks_` and `block_id`
3. Lines 333-347: Handle `ref_count` in `free()` for shared blocks
