#include <iostream>
#include "matrix_mult.h"

int main() {
    std::cout << "CUDA Matrix Multiplication Benchmark" << std::endl;

    int sizes[] = {512, 1024, 2048};
    for (int size : sizes) {
        runMatrixMulBenchmark(size);
    }

    return 0;
}
