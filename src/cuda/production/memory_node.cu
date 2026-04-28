#include "cuda/production/memory_node.h"

namespace cuda::production {

MemoryNode GraphMemoryManager::add_device_allocation(GraphExecutor& graph,
                                                     cudaGraph_t cuda_graph,
                                                     size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));

    cudaMemcpy3DParms params{};
    params.srcArray = nullptr;
    params.srcPos = make_cudaPos(0, 0, 0);
    params.srcPtr = cudaPitchedPtr{ptr, size, size, 1};
    params.dstArray = nullptr;
    params.dstPos = make_cudaPos(0, 0, 0);
    params.dstPtr = cudaPitchedPtr{ptr, size, size, 1};
    params.kind = cudaMemcpyDeviceToDevice;

    cudaGraphNode_t memcpy_node;
    CUDA_CHECK(cudaGraphAddMemcpyNode(&memcpy_node, cuda_graph, nullptr, 0, &params));

    allocations_.push_back({ptr, size, MemoryType::Device});
    total_allocated_ += size;

    return MemoryNode(memcpy_node, MemoryType::Device, ptr, size);
}

MemoryNode GraphMemoryManager::add_host_allocation(GraphExecutor& graph,
                                                   cudaGraph_t cuda_graph,
                                                   size_t size) {
    void* host_ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_ptr, size));

    void* device_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, size));

    cudaMemcpy3DParms params{};
    params.srcArray = nullptr;
    params.srcPos = make_cudaPos(0, 0, 0);
    params.srcPtr = cudaPitchedPtr{host_ptr, size, size, 1};
    params.dstArray = nullptr;
    params.dstPos = make_cudaPos(0, 0, 0);
    params.dstPtr = cudaPitchedPtr{device_ptr, size, size, 1};
    params.kind = cudaMemcpyHostToDevice;

    cudaGraphNode_t memcpy_to_device;
    CUDA_CHECK(cudaGraphAddMemcpyNode(&memcpy_to_device, cuda_graph, nullptr, 0, &params));

    allocations_.push_back({host_ptr, size, MemoryType::HostPinned});
    allocations_.push_back({device_ptr, size, MemoryType::Device});
    total_allocated_ += size * 2;

    return MemoryNode(memcpy_to_device, MemoryType::HostPinned, host_ptr, size);
}

MemoryNode GraphMemoryManager::add_managed_allocation(GraphExecutor& graph,
                                                      cudaGraph_t cuda_graph,
                                                      size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));

    cudaMemsetParams params{};
    params.dst = ptr;
    params.pitch = size;
    params.value = 0;
    params.width = size;
    params.height = 1;

    cudaGraphNode_t memset_node;
    CUDA_CHECK(cudaGraphAddMemsetNode(&memset_node, cuda_graph, nullptr, 0, &params));

    allocations_.push_back({ptr, size, MemoryType::Managed});
    total_allocated_ += size;

    return MemoryNode(memset_node, MemoryType::Managed, ptr, size);
}

void GraphMemoryManager::free_device(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
    allocations_.erase(
        std::remove_if(allocations_.begin(), allocations_.end(),
                       [ptr](const Allocation& a) { return a.ptr == ptr; }),
        allocations_.end());
}

void GraphMemoryManager::free_host(void* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
    allocations_.erase(
        std::remove_if(allocations_.begin(), allocations_.end(),
                       [ptr](const Allocation& a) { return a.ptr == ptr; }),
        allocations_.end());
}

void GraphMemoryManager::free_managed(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
    allocations_.erase(
        std::remove_if(allocations_.begin(), allocations_.end(),
                       [ptr](const Allocation& a) { return a.ptr == ptr; }),
        allocations_.end());
}

void GraphMemoryManager::cleanup() {
    for (const auto& alloc : allocations_) {
        switch (alloc.type) {
            case MemoryType::Device:
                cudaFree(alloc.ptr);
                break;
            case MemoryType::HostPinned:
                cudaFreeHost(alloc.ptr);
                break;
            case MemoryType::Managed:
                cudaFree(alloc.ptr);
                break;
        }
    }
    allocations_.clear();
    total_allocated_ = 0;
}

}  // namespace cuda::production
