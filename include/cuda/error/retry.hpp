#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <random>
#include <thread>

namespace nova::error {

enum class retry_result { success, failure, timeout };

struct retry_config {
    std::chrono::milliseconds base_delay{100};
    double multiplier{2.0};
    std::chrono::milliseconds max_delay{30000};
    int max_attempts{5};
    bool jitter_enabled{true};
};

struct circuit_breaker_config {
    int failure_threshold{5};
    std::chrono::seconds reset_timeout{30};
    int half_open_success_threshold{3};
};

enum class circuit_state { closed, open, half_open };

class circuit_breaker {
public:
    explicit circuit_breaker(circuit_breaker_config config);
    circuit_breaker(circuit_breaker&&) noexcept = default;
    circuit_breaker& operator=(circuit_breaker&&) noexcept = default;

    [[nodiscard]] bool allow_request() const;
    void record_success();
    void record_failure();
    [[nodiscard]] circuit_state state() const noexcept { return state_; }

private:
    void transition_to_open();
    void transition_to_half_open();
    void transition_to_closed();

    circuit_breaker_config config_;
    circuit_state state_{circuit_state::closed};
    int failure_count_{0};
    int success_count_{0};
    std::chrono::steady_clock::time_point last_failure_time_;
};

class retry_executor {
public:
    explicit retry_executor(retry_config config);

    template<typename Func>
    std::invoke_result_t<Func> execute(Func&& func);

    void set_circuit_breaker(circuit_breaker cb);
    [[nodiscard]] int attempt_count() const noexcept { return attempts_; }
    [[nodiscard]] bool was_successful() const noexcept { return success_; }

private:
    std::chrono::milliseconds calculate_delay(int attempt);
    std::chrono::milliseconds apply_jitter(std::chrono::milliseconds delay);

    retry_config config_;
    circuit_breaker circuit_breaker_;
    int attempts_{0};
    bool success_{false};
    mutable std::uniform_int_distribution<int> dist_;
};

inline std::chrono::milliseconds calculate_backoff(int attempt,
                                                   std::chrono::milliseconds base,
                                                   double multiplier,
                                                   std::chrono::milliseconds max_delay) {
    auto delay = static_cast<double>(base.count()) * std::pow(multiplier, attempt - 1);
    auto capped = std::min(delay, static_cast<double>(max_delay.count()));
    return std::chrono::milliseconds(static_cast<int>(capped));
}

inline std::chrono::milliseconds full_jitter(std::chrono::milliseconds delay) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, static_cast<int>(delay.count()));
    return std::chrono::milliseconds(dist(gen));
}

template<typename Func>
std::invoke_result_t<Func> retry_executor::execute(Func&& func) {
    success_ = false;
    attempts_ = 0;

    while (attempts_ < config_.max_attempts) {
        if (!circuit_breaker_.allow_request()) {
            throw std::runtime_error("Circuit breaker is open");
        }

        ++attempts_;
        try {
            auto result = func();
            circuit_breaker_.record_success();
            success_ = true;
            return result;
        } catch (...) {
            circuit_breaker_.record_failure();
            if (attempts_ < config_.max_attempts) {
                auto delay = calculate_delay(attempts_);
                if (config_.jitter_enabled) {
                    delay = apply_jitter(delay);
                }
                std::this_thread::sleep_for(delay);
            }
        }
    }

    throw std::runtime_error("Max retry attempts exceeded");
}

} // namespace nova::error
