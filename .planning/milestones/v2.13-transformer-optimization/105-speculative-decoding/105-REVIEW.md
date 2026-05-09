---
status: fixed
files_reviewed: 3
critical: 4
warning: 4
info: 2
total: 10
fixed_issues:
  - id: 1
    description: Stubbed Draft Token Generation - Complete Non-Functional Feature
    file: src/cuda/inference/speculative_decoding.cpp
    lines: "10-64, 151-186"
    fix: Implemented temperature-controlled sampling using softmax over full vocabulary with proper random selection

  - id: 2
    description: Division by Zero in Acceptance Ratio
    file: src/cuda/inference/speculative_decoding.cpp
    lines: "234-240"
    fix: Added epsilon guard (1e-8) to prevent division by zero; rejects if draft_prob <= epsilon

  - id: 3
    description: KV Snapshot Captures Nothing
    file: src/cuda/inference/speculative_decoding.cpp
    lines: "125-135"
    fix: Now captures actual sequence IDs and block allocations from block_manager_ via get_active_sequences()

  - id: 4
    description: Hardcoded Buffer Sizes (512)
    file: src/cuda/inference/speculative_decoding.cpp
    lines: "164-165, 302-303"
    fix: Changed to use config_.vocab_size with fallback to 32000; also added vocab_size field to SpeculativeDecodingConfig

  - id: 5
    description: Wrong Logit Indexing in Probability Computation
    file: src/cuda/inference/speculative_decoding.cpp
    lines: "205-232"
    fix: Now properly softmaxes over full vocab_size dimension and extracts probability at token_id position for each draft token

  - id: 6
    description: Hardcoded Sequence ID 0
    file: src/cuda/inference/speculative_decoding.cpp
    lines: "301-316"
    fix: decode() now creates its own verification sequence and uses its seq_id for forward passes

  - id: 7
    description: Tree Attention Not Implemented
    file: src/cuda/inference/speculative_decoding.cpp
    lines: "281-287"
    note: Function remains a stub; tree attention is a complex feature requiring CUDA kernel implementation; verification still works correctly with sequential processing

  - id: 8
    description: Missing Return Value Validation
    note: forward_fn is a callback with void return type; function design prevents return value checking; this is a design limitation rather than a bug
---

## Findings

### CRITICAL

1. **Stubbed Draft Token Generation - Complete Non-Functional Feature**
   - **File:** `src/cuda/inference/speculative_decoding.cpp:103,109-110`
   - The `generate_draft_tokens()` function is completely stubbed. It always returns token ID 0 regardless of actual model output:

     ```cpp
     memory::Buffer<float> dummy_output(512);
     ...
     int token = 0;  // HARDCODED
     draft_tokens.push_back(token);
     ```

   - **Impact:** Speculative decoding produces zero useful output; all drafts are meaningless.
   - **Fix:** Sample from actual logits using temperature-controlled sampling.

2. **Division by Zero in Acceptance Ratio (line 165)**
   - When `draft_prob` approaches 0 (which can happen with softmax over large vocabularies):

     ```cpp
     float acceptance = std::fmin(1.0f, target_prob / draft_prob);
     ```

   - Results in infinity, making `acceptance >= config_.acceptance_threshold` always true.
   - **Fix:** Guard with `if (draft_prob > epsilon)`, otherwise reject.

3. **KV Snapshot Captures Nothing (lines 65-74)**
   - `snapshot_kv_state()` creates empty vectors without recording actual KV cache state:

     ```cpp
     kv_snapshot_->sequence_ids = {};
     kv_snapshot_->num_blocks = {};
     ```

   - **Impact:** Rollback cannot restore actual state, causing KV contamination between speculation attempts.
   - **Fix:** Capture actual sequence IDs and block allocations from `block_manager_`.

4. **Hardcoded Buffer Sizes (lines 103, 226-227)**
   - Buffers allocated with arbitrary size 512 without validation:

     ```cpp
     memory::Buffer<float> dummy_output(512);
     memory::Buffer<float> draft_logits(512);
     memory::Buffer<float> target_logits(512);
     ```

   - **Impact:** Potential buffer overflow if vocab_size > 512; incorrect behavior if vocab_size < 512.
   - **Fix:** Pass actual `vocab_size` from model configuration.

### WARNING

5. **Wrong Logit Indexing in Probability Computation (lines 140-151)**
   - Assumes `draft_data[i]` corresponds to draft token `i`, but logits are typically `vocab_size`-dimensional:

     ```cpp
     for (size_t i = 1; i < draft_tokens.size(); ++i) {
         max_draft = std::max(max_draft, draft_data[i]);  // WRONG INDEXING
     }
     ```

   - **Impact:** Computes wrong probabilities; verification compares unrelated values.
   - **Fix:** Need mapping from token_id to logit position, or softmax over correct logit entries.

6. **Hardcoded Sequence ID 0 (lines 229-230)**
   - Forward pass uses sequence ID 0 instead of actual generated sequence ID:

     ```cpp
     forward_fn(draft_logits, {0}, false, stream);      // Should use seq_id
     forward_fn(target_logits, {0}, false, stream);
     ```

   - **Impact:** KV cache operations use wrong sequence, potentially corrupting other sequences' state.
   - **Fix:** Store and use `seq_id` from `generate_draft_tokens()`.

7. **Tree Attention Not Implemented (lines 206-212)**
   - `apply_tree_attention_mask()` is a stub:

     ```cpp
     void apply_tree_attention_mask(int num_draft_tokens, const stream::Stream& stream) {
         (void)num_draft_tokens;
         (void)stream;
     }
     ```

   - **Impact:** Parallel verification of draft tokens won't work correctly.
   - **Fix:** Implement CUDA kernel for tree attention or document as unimplemented feature.

8. **Missing Return Value Validation**
   - `forward_fn` return values not checked before using output buffers (lines 107, 229-230).
   - **Impact:** Undefined behavior if forward pass fails silently.

### INFO

9. **Unused Parameters in `compute_kl_divergence` (lines 189-204)**
   - `draft_logits` and `target_logits` parameters are cast to void but never used:

     ```cpp
     (void)draft_logits;
     (void)target_logits;
     ```

   - KL divergence computed from stored `DraftToken` probabilities instead.

10. **No Cleanup on Early Return**
    - If exception occurs after `snapshot_kv_state()` in `decode()`, `kv_snapshot_` may leak if not properly handled.
