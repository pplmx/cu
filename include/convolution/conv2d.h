#pragma once

#include <cstddef>
#include <cstdint>

template<typename T>
void convolve2D(const T* d_input, T* d_output,
                const T* d_kernel,
                size_t width, size_t height,
                int kernel_size);

void createGaussianKernel(float* d_kernel, int size, float sigma);

void createBoxKernel(float* d_kernel, int size);

void createSobelKernelX(float* d_kernel);

void createSobelKernelY(float* d_kernel);
