#include "sobel_edge.h"
#include "cuda_utils.h"
#include <math.h>

__global__ void sobelKernel(const uint8_t* input, uint8_t* output,
                            size_t width, size_t height, float threshold) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        output[y * width + x] = 0;
        return;
    }

    auto getGray = [&](int row, int col) -> float {
        size_t idx = (row * static_cast<int>(width) + col) * 3;
        float r = static_cast<float>(input[idx]);
        float g = static_cast<float>(input[idx + 1]);
        float b = static_cast<float>(input[idx + 2]);
        return 0.299f * r + 0.587f * g + 0.114f * b;
    };

    float gx = -getGray(y-1, x-1) - 2.0f*getGray(y, x-1) - getGray(y+1, x-1)
               +getGray(y-1, x+1) + 2.0f*getGray(y, x+1) + getGray(y+1, x+1);

    float gy = -getGray(y-1, x-1) - 2.0f*getGray(y-1, x) - getGray(y-1, x+1)
               +getGray(y+1, x-1) + 2.0f*getGray(y+1, x) + getGray(y+1, x+1);

    int magnitude = static_cast<int>(sqrtf(gx*gx + gy*gy));
    output[y * width + x] = (magnitude > static_cast<int>(threshold)) ? 255 : 0;
}

void sobelEdgeDetection(const uint8_t* d_input, uint8_t* d_output,
                        size_t width, size_t height,
                        float threshold) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sobelKernel<<<grid, block>>>(d_input, d_output, width, height, threshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
