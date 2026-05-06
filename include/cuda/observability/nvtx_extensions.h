#pragma once

#if defined(NOVA_NVTX_ENABLED) && NOVA_NVTX_ENABLED

#include <nvtx3/nvtx3.hpp>

#define NOVA_NVTX_SCOPED_RANGE(name) nvtx3::scoped_range name(name)
#define NOVA_NVTX_PUSH_RANGE(name) nvtx3::mark(name)
#define NOVA_NVTX_POP_RANGE() nvtx3::pop_range()

namespace cuda::observability {

using NVTXDomainHandle = void*;

struct NVTXDomains {
    static constexpr NVTXDomainHandle Memory = nullptr;
    static constexpr NVTXDomainHandle Device = nullptr;
    static constexpr NVTXDomainHandle Algo = nullptr;
    static constexpr NVTXDomainHandle API = nullptr;
    static constexpr NVTXDomainHandle Production = nullptr;
    static constexpr NVTXDomainHandle Performance = nullptr;
    static constexpr NVTXDomainHandle NVBlox = nullptr;
    static constexpr NVTXDomainHandle Fusion = nullptr;
    static constexpr NVTXDomainHandle Bandwidth = nullptr;
};

template <NVTXDomainHandle Domain>
class ScopedRange {
public:
    explicit ScopedRange(const char* name) {
        nvtx3::mark(name);
    }
};

template <NVTXDomainHandle Domain>
void push_range(const char* name) {
    nvtx3::mark(name);
}

template <NVTXDomainHandle Domain>
void pop_range() {
}

}  // namespace cuda::observability

#else

#define NOVA_NVTX_SCOPED_RANGE(name) ((void)0)
#define NOVA_NVTX_PUSH_RANGE(name) ((void)0)
#define NOVA_NVTX_POP_RANGE() ((void)0)

namespace cuda::observability {

template <typename Domain>
class ScopedRange {
public:
    explicit ScopedRange(const char*) {}
};

template <typename Domain>
void push_range(const char*) {}

template <typename Domain>
void pop_range() {}

struct NVTXDomains {
    static constexpr void* Memory = nullptr;
    static constexpr void* Device = nullptr;
    static constexpr void* Algo = nullptr;
    static constexpr void* API = nullptr;
    static constexpr void* Production = nullptr;
    static constexpr void* Performance = nullptr;
    static constexpr void* NVBlox = nullptr;
    static constexpr void* Fusion = nullptr;
    static constexpr void* Bandwidth = nullptr;
};

}  // namespace cuda::observability

#endif
