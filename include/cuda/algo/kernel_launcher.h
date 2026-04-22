#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

#include "cuda/device/error.h"

namespace cuda::detail {

    class KernelLauncher {
    public:
        KernelLauncher() = default;

        KernelLauncher& grid(dim3 g) & {
            grid_ = g;
            return *this;
        }

        KernelLauncher& block(dim3 b) & {
            block_ = b;
            return *this;
        }

        KernelLauncher& shared(size_t s) & {
            shared_ = s;
            return *this;
        }

        KernelLauncher& stream(cudaStream_t s) & {
            stream_ = s;
            return *this;
        }

        template <typename Kernel, typename... Args>
        void launch(Kernel* kernel, Args&&... args) {
            void* ptrs[] = {const_cast<std::remove_cv_t<std::remove_reference_t<Args>>*>(&args)...};
            CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid_, block_, ptrs, shared_, stream_));
            CUDA_CHECK(cudaGetLastError());
        }

        void synchronize() const {
            if (stream_) {
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            } else {
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        [[nodiscard]] dim3 get_grid() const { return grid_; }
        [[nodiscard]] dim3 get_block() const { return block_; }
        [[nodiscard]] size_t get_shared() const { return shared_; }
        [[nodiscard]] cudaStream_t get_stream() const { return stream_; }

    private:
        dim3 grid_{1, 1, 1};
        dim3 block_{1, 1, 1};
        size_t shared_{0};
        cudaStream_t stream_{nullptr};
    };

    [[nodiscard]] constexpr inline dim3 calc_grid_1d(size_t n, dim3 block = {256, 1, 1}) {
        return dim3{static_cast<unsigned int>((n + block.x - 1) / block.x), 1, 1};
    }

    [[nodiscard]] constexpr inline dim3 calc_grid_2d(size_t w, size_t h, dim3 block = {16, 16, 1}) {
        return dim3{static_cast<unsigned int>((w + block.x - 1) / block.x), static_cast<unsigned int>((h + block.y - 1) / block.y), 1};
    }

    [[nodiscard]] constexpr inline dim3 calc_grid_3d(size_t x, size_t y, size_t z, dim3 block = {8, 8, 8}) {
        return dim3{
            static_cast<unsigned int>((x + block.x - 1) / block.x), static_cast<unsigned int>((y + block.y - 1) / block.y), static_cast<unsigned int>((z + block.z - 1) / block.z)};
    }

}  // namespace cuda::detail
