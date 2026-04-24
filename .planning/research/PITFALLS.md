# Pitfalls Research: Multi-GPU CUDA Support for Nova

**Domain:** CUDA Multi-GPU Programming (Single-node, not distributed)
**Researched:** 2026-04-24
**Confidence:** HIGH (based on established CUDA programming patterns and NCCL best practices)

## Critical Pitfalls

### Pitfall 1: Peer Access Without Validation

**What goes wrong:**
Runtime failures when attempting peer-to-peer (P2P) memory access between GPUs that don't support it. Code compiles, runs on developer's 2-GPU machine, but crashes on single-GPU systems or certain PCIe topologies (e.g., GPUs on same PCIe root complex vs. different root complexes).

**Why it happens:**
Developers assume all modern GPUs support P2P access. Not checking `cudaDeviceCanAccessPeer()` before enabling peer access causes CUDA errors at runtime. Even when peer access is "supported" by the API, PCIe topology can prevent actual peer access between certain GPU pairs (e.g., GPU 0 and GPU 2 might be on different PCIe root complexes with no direct path).

**How to avoid:**
1. Always check `cudaDeviceCanAccessPeer()` before attempting peer access:
   ```cpp
   int can_access;
   CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src_device, dst_device));
   if (!can_access) {
       // Fall back to host-mediated transfer or reject the configuration
       return fallback_path();
   }
   ```
2. Query PCIe topology using `nvlink` or `nvidia-smi topo` at initialization
3. Build a peer access matrix at startup: `peer_access[gpu_i][gpu_j]` for all GPU pairs
4. Provide explicit single-GPU fallback for all multi-GPU operations

**Warning signs:**
- Code uses `cudaMemcpyPeer` or unified addressing without first checking peer capability
- `cudaErrorPeerAccessNotEnabled` or `cudaErrorInvalidDevice` runtime errors
- Works on developer's machine but fails in CI/CD (likely single-GPU CI runners)

**Phase to address:** Phase 1 (Device Mesh Detection) — MGPU-01

---

### Pitfall 2: Memory Consistency Across GPUs

**What goes wrong:**
Stale data when reading peer memory. GPU A writes to its memory, then GPU B reads from GPU A's memory via peer access, but sees old data or garbage. This manifests as intermittent correctness bugs that are extremely hard to debug.

**Why it happens:**
GPU memory is not coherent by default. Peer memory reads bypass the normal GPU memory hierarchy. CUDA stream dependencies (via events or stream ordering) are NOT sufficient for cross-GPU consistency — they only order operations within a single GPU. The write on GPU A might not be globally visible when GPU B starts its read operation.

**How to avoid:**
1. Use explicit `cudaDeviceSynchronize()` or `cudaEventSynchronize()` before peer reads that depend on peer writes
2. For cross-GPU operations, use a host thread as synchronization arbiter:
   ```cpp
   // GPU A writes
   cudaEventRecord(write_done, stream_A);
   cudaEventSynchronize(write_done);  // HOST waits
   // Now safe for GPU B to read
   cudaStreamWaitEvent(stream_B, write_done, 0);
   ```
3. For NCCL-based operations, rely on NCCL's internal synchronization (but be aware of NCCL's ordering requirements)
4. Never assume stream event ordering is sufficient for cross-GPU consistency

**Warning signs:**
- Intermittent wrong answers in multi-GPU reductions
- Data corruption that only appears with >2 GPUs
- Bug disappears when adding extra `cudaDeviceSynchronize()` calls

**Phase to address:** Phase 1 (Device Mesh) + Phase 2 (Data Parallelism Primitives)

---

### Pitfall 3: Synchronization Deadlocks

**What goes wrong:**
Program hangs indefinitely. Common scenario: GPU 0 waits for GPU 1, GPU 1 waits for GPU 0, both blocked.

**Why it happens:**
1. Circular dependencies: GPU 0 depends on GPU 1's result, GPU 1 depends on GPU 0's result
2. Mismatched collective operations: One GPU enters all-reduce, another doesn't
3. Using blocking operations (like `cudaMemcpy` peer) in streams that other streams wait on
4. NCCL communicators not properly initialized before collective calls

**How to avoid:**
1. Design communication patterns as directed acyclic graphs (DAGs), never bidirectional
2. For bidirectional data exchange, use non-blocking APIs with explicit completion events:
   ```cpp
   // Instead of blocking:
   // cudaMemcpyPeer(dst, dst_dev, src, src_dev, size);  // BLOCKING - DANGEROUS
   
   // Use non-blocking with explicit ordering:
   cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);  // OK if streams properly ordered
   ```
3. Verify all GPUs participate in collective operations — use consistent grid dimensions
4. Implement deadlock detection in test suite (time out operations, assert on hang)
5. Use `cudaStreamQuery()` periodically to detect stuck streams

**Warning signs:**
- Program hangs without error message
- `nvidia-smi` shows GPU utilization stuck at 100% with no progress
- Deadlock appears only with specific GPU counts (2 vs 4)

**Phase to address:** Phase 2 (Data Parallelism Primitives) — MGPU-02

---

### Pitfall 4: Ignoring NVLink vs PCIe Topology

**What goes wrong:**
Poor multi-GPU performance because data is transferred over slow PCIe buses instead of fast NVLink. Or: code works but performance is 10x worse than expected.

**Why it happens:**
NVLink provides ~300 GB/s bidirectional bandwidth. PCIe Gen4 x16 provides ~32 GB/s. For large tensor parallelism or frequent cross-GPU communication, this difference dominates performance. Many developers treat all peer access as "equally fast" or assume NVLink exists everywhere.

**How to avoid:**
1. Detect topology at initialization:
   ```cpp
   // Check NVLink connections
   int nvlink_count = 0;
   for (int i = 0; i < num_gpus; i++) {
       for (int j = i + 1; j < num_gpus; j++) {
           if (has_nvlink(i, j)) nvlink_count++;
       }
   }
   ```
2. Optimize data placement for topology — put frequently communicating tensors on NVLink-connected GPUs
3. Use `cudaGetDeviceProperties` with `pciBusId` to understand topology
4. Implement bandwidth-aware scheduling: prefer NVLink paths over PCIe
5. Document expected topology requirements for optimal performance

**Warning signs:**
- Multi-GPU matmul 2-10x slower than single-GPU
- All-reduce takes disproportionately long
- `nvidia-smi` shows low GPU utilization but high PCIe traffic

**Phase to address:** Phase 1 (Device Mesh Detection) + Phase 4 (Multi-GPU Matmul) — MGPU-01, MGPU-04

---

### Pitfall 5: NCCL Initialization Race Conditions

**What goes wrong:**
NCCL communicators fail to initialize, or initialized incorrectly, causing crashes or wrong results. Common: "NCCL error: unhandled system error" or hangs during init.

**Why it happens:**
1. NCCL requires one communicator per unique (process, local rank, CUDA device) tuple
2. Calling `ncclCommInitAll` from multiple threads simultaneously causes races
3. NCCL communicators must be destroyed before CUDA context is destroyed (order matters)
4. Memory leaks from unfreed NCCL communicators on repeated initialization/teardown

**How to avoid:**
1. Initialize NCCL in a controlled singleton pattern with mutex protection:
   ```cpp
   class NCCLContext {
       static std::mutex init_mutex;
       static std::unique_ptr<NCCLContext> instance;
       
   public:
       static NCCLContext& get() {
           std::lock_guard<std::mutex> lock(init_mutex);
           if (!instance) {
               instance = std::make_unique<NCCLContext>();
               instance->init();
           }
           return *instance;
       }
       
       ~NCCLContext() {
           for (auto& comm : communicators_) {
               ncclCommDestroy(comm);
           }
       }
   };
   ```
2. Ensure NCCL communicator destruction happens BEFORE CUDA context cleanup
3. Track communicator lifecycle explicitly — log creation and destruction
4. Handle `ncclSystemError` by checking NCCL version compatibility

**Warning signs:**
- "NCCL error: unhandled system error" or "internal error" at startup
- Memory grows on repeated multi-GPU operations (leaked communicators)
- Hangs during `ncclGroupStart()` / `ncclGroupEnd()` block

**Phase to address:** Phase 1 (Device Mesh Detection) if NCCL-based, or Phase 2 (Data Parallelism)

---

### Pitfall 6: Single-GPU Fallback Not Working

**What goes wrong:**
Multi-GPU code compiles but crashes or produces wrong results when only 1 GPU is present. Code assumes multiple GPUs exist and accesses out-of-bounds indices.

**Why it happens:**
1. Array indexing assumes `device_id < num_devices` but num_devices is 1
2. Default device selection assumes 0 < num_devices when it could equal 1
3. Loop `for (int i = 0; i < num_gpus; i++)` where `num_gpus = 0` causes issues
4. Building peer access matrix with 0-size allocation

**How to avoid:**
1. Design all multi-GPU primitives to have a single-GPU fast path:
   ```cpp
   template<typename Op>
   auto multi_gpu_reduce(const Tensor& input, Op op) -> Tensor {
       int num_gpus = get_gpu_count();
       if (num_gpus == 1) {
           return single_gpu_reduce(input, op);  // Bypass all multi-GPU code
       }
       // Multi-GPU path...
   }
   ```
2. Test on single-GPU systems — CI/CD must include single-GPU runner
3. Validate `num_gpus > 1` guards at API entry points
4. Ensure build works without `NCCL` dependency (make it optional)

**Warning signs:**
- Code crashes immediately on single-GPU systems
- "vector subscript out of range" when initializing device mesh
- `std::bad_alloc` from negative-size allocations

**Phase to address:** Phase 1 (Device Mesh Detection) — foundation requirement, tested throughout

---

### Pitfall 7: Tensor Parallelism Communication Patterns

**What goes wrong:**
Multi-GPU matmul produces wrong results. Values aregarbage, NaN, or incorrect by exactly half (missing all-reduce). Or: correct results but 2x slower than expected (duplicate all-reduce).

**Why it happens:**
1. Column-parallel matmul: Each GPU holds columns of weight matrix W. Output has partial results that must be all-reduced.
2. Row-parallel matmul: Each GPU holds rows of input X. Need all-gather before computation.
3. Forgetting the required communication after matmul for column-parallel, or before for row-parallel.
4. Inverting the communication pattern (all-reduce instead of all-gather or vice versa).

**How to avoid:**
1. Document the parallelism strategy explicitly:
   ```
   Column-Parallel matmul (Y = X @ W):
   - X: replicated across GPUs
   - W: column-partitioned (each GPU has columns [i..j])
   - Y: each GPU has partial columns — requires ALL-REDUCE at output
   
   Row-Parallel matmul (Y = X @ W):
   - X: row-partitioned (each GPU has rows [i..j])
   - W: replicated across GPUs
   - Before: ALL-GATHER to reconstruct full X
   - After: no communication needed (each GPU has its output rows)
   ```
2. Implement verification kernels that check result norms match single-GPU baseline
3. Test with known-good single-GPU reference implementation
4. Use assertion-based checking in debug builds:
   ```cpp
   #ifdef DEBUG
   auto expected = single_gpu_matmul(input, weight);
   auto actual = multi_gpu_matmul(input, weight);
   ASSERT_NEAR(expected.norm(), actual.norm(), tolerance);
   #endif
   ```

**Warning signs:**
- Output values are exactly half or double expected
- NaN values appearing in specific GPU configurations
- Gradient checks failing in multi-GPU training

**Phase to address:** Phase 4 (Multi-GPU Matmul) — MGPU-04

---

### Pitfall 8: Stream Per Device Without Proper Scope

**What goes wrong:**
Operations on GPU 0 use streams created on GPU 0, but operations that depend on them on GPU 1 can't wait on those streams directly. Stream dependencies are per-device.

**Why it happens:**
CUDA streams are device-specific. `cudaStreamWaitEvent` only works when the stream and event are on the same device. Creating a stream on GPU 0 and passing it to operations on GPU 1 causes CUDA errors.

**How to avoid:**
1. Maintain a `std::vector<cudaStream_t> streams_per_gpu` structure
2. For cross-GPU dependencies, use events recorded on source GPU and waited on by target GPU's stream:
   ```cpp
   // GPU 0 does work and records event
   cudaSetDevice(0);
   cudaEventRecord(done_event_0_to_1, stream_0);
   
   // GPU 1 waits for GPU 0's event
   cudaSetDevice(1);
   cudaStreamWaitEvent(stream_1, done_event_0_to_1, 0);
   ```
3. Use a device-scoped RAII wrapper for stream management:
   ```cpp
   class DeviceStreams {
       std::vector<std::vector<cudaStream_t>> streams_;  // [device][streams]
   public:
       cudaStream_t get_stream(int device, int stream_idx);
   };
   ```

**Warning signs:**
- `cudaErrorInvalidResourceHandle` when waiting on streams
- Cross-GPU synchronization appears to work but events aren't actually ordering operations
- Race conditions in multi-threaded multi-GPU code

**Phase to address:** Phase 1 (Device Mesh) — this is a foundational issue

---

### Pitfall 9: Distributed Memory Pool Without Coherency

**What goes wrong:**
Memory allocated from GPU 1's memory pool is accessed by GPU 0, but GPU 0's memory pool doesn't track it. When "deallocate" is called, GPU 0's pool doesn't know about the allocation, causing memory leaks or double-free.

**Why it happens:**
The existing Nova `MemoryPool` is per-device (no device ID). Extending to multi-GPU requires either:
1. A global pool that tracks allocations across devices (complex)
2. Per-device pools with explicit cross-device allocation tracking (simpler)

Most developers implement option 1 incorrectly by not tracking which device owns which memory.

**How to avoid:**
1. Design cross-device memory tracking:
   ```cpp
   struct CrossDeviceAllocation {
       void* ptr;
       int owning_device;  // Which GPU owns this memory
       size_t bytes;
       int requesting_device;  // Which GPU requested it
   };
   
   class DistributedMemoryPool {
       std::vector<MemoryPool> per_device_pools_;
       std::unordered_map<void*, CrossDeviceAllocation> cross_device_allocations_;
   public:
       void* allocate(size_t bytes, int requesting_device, int owning_device);
       void deallocate(void* ptr);
   };
   ```
2. Prefer local allocations: if GPU i needs memory, allocate on GPU i
3. Only use peer access for temporary cross-GPU data, not persistent allocations
4. Ensure destructor order: destroy cross-device allocations BEFORE destroying individual device pools

**Warning signs:**
- Memory leaks detected by cuda-memcheck on multi-GPU workloads
- "invalid device pointer" errors when freeing cross-GPU allocations
- Memory usage grows unbounded with iteration count

**Phase to address:** Phase 3 (Distributed Memory Pool) — MGPU-03

---

### Pitfall 10: Not Testing Error Paths

**What goes wrong:**
Multi-GPU code only tested in the happy path. Production failures happen when:
- GPU dies mid-computation
- PCIe link degrades
- One GPU in a 4-GPU system fails to initialize
- NCCL times out waiting for straggler

**Why it happens:**
Multi-GPU code has many more failure modes than single-GPU. If code only handles success paths, any failure causes crashes or hangs.

**How to avoid:**
1. Add chaos testing: randomly fail GPU operations in test suite
2. Test with `CUDA_VISIBLE_DEVICES=""` to simulate no GPUs
3. Test with subset of GPUs visible (simulate partial failure)
4. Add timeout wrappers around all collective operations:
   ```cpp
   template<typename F>
   ncclResult_t timed_nccl_call(F&& fn, int timeout_ms) {
       std::atomic<bool> timed_out(false);
       std::thread timeout_thread([&]() {
           std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
           timed_out = true;
       });
       
       ncclResult_t result = fn();
       timeout_thread.join();
       
       if (timed_out) {
           return ncclUnhandledCudaError;
       }
       return result;
   }
   ```
5. Implement graceful degradation: if N of M GPUs available, split work among N

**Warning signs:**
- Test suite passes but production has occasional hangs
- No timeout on NCCL operations
- Single point of failure in device enumeration

**Phase to address:** All phases — add error testing throughout

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip peer access validation | Faster dev, assumes modern GPUs | Crash on legacy/single-GPU systems | Never in production code |
| Skip NCCL error checking | Simpler code | Silent failures, corrupted results | Never |
| Hardcode num_gpus = 2 | Simpler indexing | Complete rewrite for other configs | Only in exploratory prototypes |
| Use blocking peer copy | Simpler code | Deadlock risk | Never in multi-stream code |
| Skip topology detection | Faster init | 10x slower on PCIe-only systems | Only when NVLink guaranteed |
| Skip single-GPU fallback | Simpler dispatch | Breaks on CI/single-GPU dev machines | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| NCCL | Calling `ncclGroupStart/End` from multiple threads | Single-threaded initialization with mutex |
| CUDA Streams | Using stream from wrong device | Maintain per-device stream vectors |
| cuBLAS | Creating handle on wrong device | Create handle after `cudaSetDevice` |
| Memory Pool | Cross-device allocation without tracking | Explicit ownership tracking or local-only allocation |
| Peer Access | Enabling peer access without validation | Always check `cudaDeviceCanAccessPeer` first |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| All-reduce on PCIe | 100x slower than expected | Check topology, use tree reductions | Always on PCIe-only systems |
| Uncoalesced cross-GPU copy | Poor bandwidth utilization | Align data layout to PCIe bus width | When transferring small tensors frequently |
| Fine-grained synchronization | GPU idle time between ops | Batch communication, overlap with compute | Every multi-GPU workload |
| Oversubscribing GPUs | OOM errors, thrashing | Respect `CUDA_VISIBLE_DEVICES`, cap concurrency | When running multiple processes per GPU |

---

## Security Mistakes

Multi-GPU CUDA code has limited traditional security concerns, but these matter:

| Mistake | Risk | Prevention |
|---------|------|------------|
| Not validating device IDs | Out-of-bounds memory access | Bounds check all device indices |
| Unvalidated peer access enablement | Crash or hang on unsupported configs | Always validate before enabling |
| Not handling CUDA errors | Silent corruption, security issues | Never ignore CUDA_CHECK return values |
| Resource exhaustion (communicators, streams) | OOM, process termination | Set hard limits, implement cleanup |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No single-GPU fallback | Code doesn't work on most dev machines | Explicit single-GPU path |
| Crashes without helpful error | Confusing failures | Detect issues early, print actionable message |
| Silent performance degradation | Slow code users don't understand | Warn if NVLink not available, suggest topology |
| No device enumeration utility | Users don't know GPU config | Provide `nova::device::enumerate_devices()` |

---

## "Looks Done But Isn't" Checklist

- [ ] **Peer Access:** Often missing validation — verify `cudaDeviceCanAccessPeer` called before every P2P op
- [ ] **Cross-GPU Sync:** Often missing explicit barriers — verify events used for cross-GPU ordering, not just streams
- [ ] **Single-GPU Fallback:** Often untested — verify code compiles and runs correctly with `CUDA_VISIBLE_DEVICES=""` or 1 GPU
- [ ] **NCCL Cleanup:** Often leaks communicators — verify valgrind/cuda-memcheck shows no leaks
- [ ] **Tensor Parallelism:** Often missing communication — verify all-reduce present after column-parallel matmul
- [ ] **Memory Pool:** Often doesn't track cross-device allocations — verify distributed pool tracks ownership
- [ ] **Topology Awareness:** Often assumed NVLink exists — verify fallback to PCIe performance is acceptable
- [ ] **Error Handling:** Often only tested in happy path — verify error paths tested in unit suite

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Peer access crash | HIGH | Add validation, re-test all configurations |
| Deadlock | HIGH | Add deadlock detection, redesign communication pattern |
| Memory leak (NCCL) | MEDIUM | Find leak with cuda-memcheck, add destructor ordering |
| Wrong tensor parallelism results | HIGH | Add verification kernels, test against single-GPU baseline |
| Performance on PCIe | LOW | Accept or invest in topology-aware scheduling |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Peer Access Without Validation | Phase 1: Device Mesh Detection (MGPU-01) | Test on single-GPU, test with restricted CUDA_VISIBLE_DEVICES |
| Memory Consistency | Phase 1: Device Mesh + Phase 2: Data Parallelism | Race detection tools, memory sanitizer |
| Synchronization Deadlocks | Phase 2: Data Parallelism Primitives (MGPU-02) | Timeout-based deadlock detection in tests |
| NVLink vs PCIe Topology | Phase 1: Device Mesh | Benchmark with nvidia-smi topo, warn users |
| NCCL Initialization Races | Phase 1 or Phase 2 (NCCL integration) | Stress test init/teardown cycles |
| Single-GPU Fallback | Phase 1: Device Mesh (foundation) | CI/CD on single-GPU runner |
| Tensor Parallelism Patterns | Phase 4: Multi-GPU Matmul (MGPU-04) | Verify against single-GPU reference |
| Stream Per Device Scope | Phase 1: Device Mesh | Code review, integration tests |
| Distributed Memory Pool Coherency | Phase 3: Distributed Memory Pool (MGPU-03) | Memory tracking tests, cuda-memcheck |
| Error Path Testing | All phases | Chaos testing, timeout wrappers |

---

## Sources

- NVIDIA CUDA Documentation: Peer-to-Peer Memory Access
- NCCL GitHub: Initialization and Communicator Management
- CUDA Best Practices Guide: Multi-GPU Programming
- Stack Overflow / NVIDIA Forums: Common Multi-GPU Pitfalls
- PyTorch Distributed: Lessons Learned from NCCL Integration
- Community post-mortems: Multi-GPU deadlock and correctness issues

---

*Pitfalls research for: Nova CUDA Library v1.1 Multi-GPU Support*
*Researched: 2026-04-24*
