#ifndef NOVA_CUDA_SPARSE_NVTX_SPARSE_HPP
#define NOVA_CUDA_SPARSE_NVTX_SPARSE_HPP

#ifdef NOVA_NVTX

#include <nvtx3/nvtx3.hpp>

namespace nova {
namespace sparse {
namespace nvtx {

inline nvtxDomainHandle_t get_sparse_domain() {
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("nova_sparse");
    return domain;
}

struct ScopedRange {
    ScopedRange(const char* name) {
        nvtxRangePushA(name);
    }
    ~ScopedRange() {
        nvtxRangePop();
    }
};

#define NOVA_NVTX_SCOPED_RANGE(name) nova::sparse::nvtx::ScopedRange nova_nvtx_range_(name)
#define NOVA_NVTX_MARKER(name) nvtxMarkA(name)

}
}

#else

namespace nova {
namespace sparse {
namespace nvtx {
inline void* get_sparse_domain() { return nullptr; }
}
}

#define NOVA_NVTX_SCOPED_RANGE(name) ((void)0)
#define NOVA_NVTX_MARKER(name) ((void)0)

#endif

#endif
