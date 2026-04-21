#include <iostream>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include "brightness.h"
#include "gaussian_blur.h"
#include "sobel_edge.h"
#include "image_utils.h"
#include "test_patterns.cuh"
#include "cuda_utils.h"

class Timer {
public:
    Timer(const char* name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(end - start_).count();
        std::cout << name_ << ": " << ms << " ms" << std::endl;
    }

private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

void runDemo(size_t width, size_t height, const char* pattern) {
    std::cout << "\n=== Demo: " << pattern << " (" << width << "x" << height << ") ===" << std::endl;

    size_t size = width * height * 3;
    std::vector<unsigned char> input(size);
    std::vector<unsigned char> output(size);

    if (strcmp(pattern, "checkerboard") == 0) {
        generateCheckerboard(input.data(), width, height, 16);
    } else if (strcmp(pattern, "gradient") == 0) {
        generateGradient(input.data(), width, height);
    } else {
        generateSolid(input.data(), width, height, 128);
    }

    uint8_t *d_input = nullptr;
    uint8_t *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice));

    {
        Timer t("Brightness/Contrast (alpha=1.5, beta=30)");
        adjustBrightnessContrast(d_input, d_output, width, height, 1.5f, 30.0f);
    }

    {
        Timer t("Gaussian Blur (sigma=2.0, size=5)");
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice));
        gaussianBlur(d_input, d_output, width, height, 2.0f, 5);
    }

    {
        Timer t("Sobel Edge Detection (threshold=50)");
        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice));
        sobelEdgeDetection(d_input, d_output, width, height, 50.0f);
    }

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    int nonZero = 0;
    for (size_t i = 0; i < size; i += 3) {
        if (output[i] > 0) nonZero++;
    }
    std::cout << "  Edge pixels: " << nonZero << std::endl;
}

int main() {
    std::cout << "=== CUDA Image Filters Demo ===" << std::endl;
    std::cout << "Demonstrating: Brightness/Contrast, Gaussian Blur, Sobel Edge Detection" << std::endl;

    runDemo(256, 256, "solid");
    runDemo(512, 512, "checkerboard");
    runDemo(1024, 1024, "gradient");

    std::cout << "\nDemo complete!" << std::endl;
    return 0;
}
