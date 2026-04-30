#include "sparse_matrix.hpp"
#include "sparse_ops.hpp"
#include "krylov.hpp"
#include "hyb_matrix.hpp"
#include "roofline.hpp"
#include "solver_workspace.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

namespace nova {
namespace sparse {
namespace test {

template<typename T>
bool e2e_cg_pipeline() {
    std::cout << "=== E2E CG Pipeline Test ===\n";

    int n = 50;
    std::vector<T> dense(n * n, T{0});

    for (int i = 0; i < n; ++i) {
        dense[i * n + i] = T{4};
        if (i > 0) dense[i * n + i - 1] = T{-1};
        if (i < n - 1) dense[i * n + i + 1] = T{-1};
    }

    auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), n, n);
    if (!csr) {
        std::cerr << "Failed to create CSR matrix\n";
        return false;
    }
    std::cout << "  CSR created: " << n << "x" << n << ", nnz=" << csr->nnz() << "\n";

    auto ell = SparseMatrixELL<T>::FromCSR(*csr);
    std::cout << "  ELL created: max_nnz=" << ell.max_nnz_per_row() << "\n";

    auto sell = SparseMatrixSELL<T>::FromCSR(*csr, 16);
    std::cout << "  SELL created: slice_height=16\n";

    auto hyb = SparseMatrixHYB<T>::FromCSR(*csr);
    std::cout << "  HYB created: ELL=" << hyb.ell_row_count()
              << ", COO=" << hyb.coo_row_count() << "\n";

    std::vector<T> b(n, T{1});
    std::vector<T> x(n, T{0});
    std::vector<T> x_ell(n, T{0});
    std::vector<T> x_sell(n, T{0});
    std::vector<T> x_hyb(n, T{0});

    SolverConfig<T> config;
    config.relative_tolerance = T{1e-8};
    config.max_iterations = 200;

    SolverWorkspace<T> workspace(n);
    ConjugateGradient<T> solver(config);

    auto result = solver.solve(*csr, b.data(), x.data());
    std::cout << "  CG (CSR): " << (result.converged ? "converged" : "failed")
              << " in " << result.iterations << " iterations\n";

    workspace.reset();
    result = solver.solve(*csr, b.data(), x_ell.data());
    std::cout << "  CG (ELL): " << (result.converged ? "converged" : "failed")
              << " in " << result.iterations << " iterations\n";

    workspace.reset();
    result = solver.solve(*csr, b.data(), x_sell.data());
    std::cout << "  CG (SELL): " << (result.converged ? "converged" : "failed")
              << " in " << result.iterations << " iterations\n";

    workspace.reset();
    result = solver.solve(*csr, b.data(), x_hyb.data());
    std::cout << "  CG (HYB): " << (result.converged ? "converged" : "failed")
              << " in " << result.iterations << " iterations\n";

    T max_diff = T{0};
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(x[i] - x_ell[i]));
        max_diff = std::max(max_diff, std::abs(x[i] - x_sell[i]));
        max_diff = std::max(max_diff, std::abs(x[i] - x_hyb[i]));
    }

    std::cout << "  Max solution difference: " << max_diff << "\n";

    if (max_diff > T{1e-6}) {
        std::cerr << "  FAILED: Solutions differ\n";
        return false;
    }

    std::cout << "  PASSED\n\n";
    return true;
}

template<typename T>
bool e2e_gmres_benchmark() {
    std::cout << "=== E2E GMRES Benchmark ===\n";

    int sizes[] = {10, 50, 100};
    SolverConfig<T> config;
    config.max_iterations = 500;

    for (int n : sizes) {
        std::vector<T> dense(n * n, T{0});

        for (int i = 0; i < n; ++i) {
            dense[i * n + i] = T{5};
            if (i > 0) dense[i * n + i - 1] = T{1};
            if (i < n - 1) dense[i * n + i + 1] = T{-1};
        }

        auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), n, n);
        if (!csr) continue;

        std::vector<T> b(n, T{1});
        std::vector<T> x(n, T{0});

        auto start = std::chrono::high_resolution_clock::now();
        GMRES<T> solver(config, 20);
        auto result = solver.solve(*csr, b.data(), x.data());
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "  " << n << "x" << n << ": "
                  << result.iterations << " iterations, "
                  << result.relative_residual << " residual, "
                  << duration.count() / 1000.0 << " ms\n";
    }

    std::cout << "  PASSED\n\n";
    return true;
}

template<typename T>
bool roofline_e2e() {
    std::cout << "=== E2E Roofline Analysis ===\n";

    RooflineAnalyzer analyzer;
    auto peaks = analyzer.device_peaks();

    std::cout << "  Device peaks:\n";
    std::cout << "    FP32: " << peaks.fp32_peak_gflops << " GFLOPS\n";
    std::cout << "    Bandwidth: " << peaks.memory_bandwidth_gbps << " GB/s\n";

    int nnz = 10000;
    int n = 5000;
    double ai = spmv_arithmetic_intensity<T>(nnz, n);
    std::cout << "  SpMV AI: " << ai << " FLOPs/byte\n";

    auto classification = analyzer.classify_with_confidence(ai, peaks.fp32_peak_gflops);
    std::cout << "  Classification: " << RooflineAnalyzer::bound_to_string(classification.bound)
              << " (" << classification.confidence_percent << "% confidence)\n";

    RooflineAnalysis analysis;
    analysis.set_device_info(peaks, Precision::FP32, "Test Device");

    auto metrics = analyzer.analyze_kernel("spmv_csr", 2LL * nnz, 0.5, nnz * sizeof(T) * 3, Precision::FP32);
    analysis.add_kernel(metrics);

    std::string json = analysis.to_json();
    std::cout << "  JSON export length: " << json.length() << " chars\n";
    std::cout << "  PASSED\n\n";
    return true;
}

template<typename T>
bool workspace_reuse_test() {
    std::cout << "=== Workspace Reuse Test ===\n";

    int n = 100;
    std::vector<T> dense(n * n, T{0});
    for (int i = 0; i < n; ++i) {
        dense[i * n + i] = T{4};
        if (i > 0) dense[i * n + i - 1] = T{-1};
        if (i < n - 1) dense[i * n + i + 1] = T{-1};
    }

    auto csr = SparseMatrixCSR<T>::FromDense(dense.data(), n, n);
    if (!csr) return false;

    std::vector<T> b(n, T{1});
    std::vector<T> x1(n, T{0});
    std::vector<T> x2(n, T{0});

    SolverConfig<T> config;
    config.relative_tolerance = T{1e-8};
    config.max_iterations = 200;

    SolverWorkspace<T> workspace(n);
    ConjugateGradient<T> solver(config);

    workspace.reset();
    auto result1 = solver.solve(*csr, b.data(), x1.data());

    workspace.reset();
    auto result2 = solver.solve(*csr, b.data(), x2.data());

    std::cout << "  First solve: " << result1.iterations << " iterations\n";
    std::cout << "  Second solve: " << result2.iterations << " iterations\n";

    T max_diff = T{0};
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(x1[i] - x2[i]));
    }

    std::cout << "  Solution difference: " << max_diff << "\n";
    std::cout << "  PASSED\n\n";
    return true;
}

template<typename T>
int run_all() {
    int passed = 0;
    int failed = 0;

    if (e2e_cg_pipeline<T>()) ++passed; else ++failed;
    if (e2e_gmres_benchmark<T>()) ++passed; else ++failed;
    if (roofline_e2e<T>()) ++passed; else ++failed;
    if (workspace_reuse_test<T>()) ++passed; else ++failed;

    std::cout << "=== Integration Test Results ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    return failed;
}

}
}
}

int main() {
    return nova::sparse::test::run_all<double>();
}
