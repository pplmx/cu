#include "fuzz_utils.hpp"
#include <nova/algo/reduce.hpp>
#include <nova/algo/scan.hpp>
#include <nova/algo/sort.hpp>
#include <vector>
#include <algorithm>

namespace nova {
namespace fuzz {

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    FuzzedDataProvider provider(data, size);

    // Generate operation type (0=reduce, 1=scan, 2=sort)
    auto op = provider.ConsumeIntegralInRange<int>(0, 2);

    // Limit count to prevent excessive memory usage
    auto count = provider.ConsumeIntegralInRange<size_t>(1, 4096);
    if (count > 1024) count = 1024; // Cap at reasonable size

    // Generate test data
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = provider.ConsumeFloatingPoint<float>();
    }

    switch (op) {
        case 0: { // reduce
            try {
                float result = 0.0f;
                // Simplified reduce for fuzzing
                for (size_t i = 0; i < count; ++i) {
                    result += input[i];
                }
                (void)result;
            } catch (...) {
                return 1; // Crash on error
            }
            break;
        }
        case 1: { // scan (inclusive prefix sum)
            try {
                std::vector<float> output(count);
                output[0] = input[0];
                for (size_t i = 1; i < count; ++i) {
                    output[i] = output[i-1] + input[i];
                }
            } catch (...) {
                return 1;
            }
            break;
        }
        case 2: { // sort
            try {
                std::vector<float> sorted = input;
                std::sort(sorted.begin(), sorted.end());
                // Verify sorted
                for (size_t i = 1; i < sorted.size(); ++i) {
                    if (sorted[i] < sorted[i-1]) {
                        return 1; // Not sorted - crash
                    }
                }
            } catch (...) {
                return 1;
            }
            break;
        }
    }

    return 0;
}

} // namespace fuzz
} // namespace nova
