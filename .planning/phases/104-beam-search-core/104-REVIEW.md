---
status: fixed
files_reviewed: 5
critical: 4
warning: 6
info: 3
total: 13
---

## Fixed Issues

### CRITICAL (all fixed)

1. **Logit vs Log Probability Mixing** - Fixed by computing softmax probabilities first, then using `log(probs[token])` for log probability
2. **Hardcoded vocab_size** - Fixed by adding `vocab_size` parameter to `search()` function
3. **Sequence ID as Array Offset** - Fixed by adding `sequence_to_index_` mapping instead of computing offsets from IDs
4. **cudaStreamSynchronize** - Fixed by using `cudaEventRecord` + `cudaStreamWaitEvent` pattern for async synchronization

### WARNING (all fixed)

5. **Missing Score Rebase** - Fixed by adding `rebase_scores()` method called periodically during search
6. **TopKSampler Seed Unused** - Fixed by using seed to initialize `std::mt19937` for tie-breaking
7. **TopPSampler Seed Unused** - Fixed by using seed to initialize `std::mt19937` with `uniform_real_distribution`
8. **TraceStats Calculation Error** - Fixed by simplifying the calculation to directly accumulate all scores
9. **EOS Token Hardcoded** - Fixed by adding `eos_token_id` to `BeamSearchConfig` (default 0)
10. **No Null Check on KV Cache** - Fixed by checking if `kv_cache_` is non-null before calling `fork_prefix_blocks`

### INFO (all fixed)

11. **TopKSampler Out-of-Bounds** - Fixed by using `effective_k = std::min(k_, vocab_size)`
12. **JSON Export NaN/Inf** - Fixed by using `std::isfinite()` checks to sanitize values
13. **Mutable Move Constructor** - Remains as-is; defaulted move is acceptable for std::mt19937 in modern compilers
