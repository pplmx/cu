#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>

#include "reduce.h"
#include "scan.h"
#include "sort.h"
#include "cuda_utils.h"

class Timer {
public:
    Timer(const char* name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<float, std::milli>(end - start_).count();
        std::cout << std::left << std::setw(35) << name_
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3)
                  << ms << " ms" << std::endl;
    }

private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

template<typename T>
void printResult(const char* name, T result) {
    std::cout << std::left << std::setw(35) << name << ": " << result << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   CUDA Parallel Algorithms Benchmark   " << std::endl;
    std::cout << "========================================" << std::endl;

    constexpr size_t N = 1 << 20;  // 1M elements

    std::cout << "\n--- Data Setup ---" << std::endl;
    std::cout << "Array size: " << N << " elements" << std::endl;

    std::vector<int> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i] = static_cast<int>(i + 1);
    }

    int *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "\n--- Reduce (Sum) ---" << std::endl;
    std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    int result = 0;
    {
        Timer t("Reduce Basic");
        result = reduceSum<int>(d_input, N);
    }
    printResult("  Sum result", result);

    {
        Timer t("Reduce Optimized");
        result = reduceSumOptimized<int>(d_input, N);
    }
    printResult("  Sum result", result);

    int maxResult = 0;
    {
        Timer t("Reduce Max");
        maxResult = reduceMax<int>(d_input, N);
    }
    printResult("  Max result", maxResult);

    std::cout << "\n--- Scan (Prefix Sum) ---" << std::endl;
    std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    constexpr size_t SCAN_SIZE = 1024;
    std::vector<int> scanInput(SCAN_SIZE);
    for (size_t i = 0; i < SCAN_SIZE; ++i) scanInput[i] = static_cast<int>(i + 1);
    std::vector<int> scanOutput(SCAN_SIZE);

    int *d_scanInput, *d_scanOutput;
    CUDA_CHECK(cudaMalloc(&d_scanInput, SCAN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scanOutput, SCAN_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_scanInput, scanInput.data(), SCAN_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    {
        Timer t("Exclusive Scan Basic");
        exclusiveScan<int>(d_scanInput, d_scanOutput, SCAN_SIZE);
    }

    {
        Timer t("Exclusive Scan Optimized");
        exclusiveScanOptimized<int>(d_scanInput, d_scanOutput, SCAN_SIZE);
    }

    CUDA_CHECK(cudaMemcpy(scanOutput.data(), d_scanOutput, SCAN_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << std::left << std::setw(35) << "  Last prefix sum" << ": " << scanOutput.back() << std::endl;

    std::cout << "\n--- Sort (Bitonic) ---" << std::endl;
    std::cout << std::left << std::setw(35) << "Algorithm" << std::right << std::setw(15) << "Time" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    constexpr size_t SORT_SIZE = 1024;
    std::vector<int> sortInput(SORT_SIZE);
    std::vector<int> sortOutput(SORT_SIZE);
    for (size_t i = 0; i < SORT_SIZE; ++i) {
        sortInput[i] = static_cast<int>((i * 17 + 31) % 1000);
    }

    int *d_sortInput, *d_sortOutput;
    CUDA_CHECK(cudaMalloc(&d_sortInput, SORT_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sortOutput, SORT_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sortInput, sortInput.data(), SORT_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    {
        Timer t("Odd-Even Sort");
        oddEvenSort<int>(d_sortInput, d_sortOutput, SORT_SIZE);
    }

    {
        Timer t("Bitonic Sort");
        bitonicSort<int>(d_sortInput, d_sortOutput, SORT_SIZE);
    }

    CUDA_CHECK(cudaMemcpy(sortOutput.data(), d_sortOutput, SORT_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    bool isSorted = std::is_sorted(sortOutput.begin(), sortOutput.end());
    std::cout << std::left << std::setw(35) << "  Sorted correctly" << ": " << (isSorted ? "YES" : "NO") << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_scanInput));
    CUDA_CHECK(cudaFree(d_scanOutput));
    CUDA_CHECK(cudaFree(d_sortInput));
    CUDA_CHECK(cudaFree(d_sortOutput));

    std::cout << "\n========================================" << std::endl;
    std::cout << "         Benchmark Complete!            " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
