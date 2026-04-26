#pragma once

#include <functional>
#include <atomic>
#include <chrono>
#include <thread>
#include <optional>
#include <cuda_runtime.h>

#if NOVA_NCCL_ENABLED
#include <nccl.h>
#endif

namespace nova::comm {

enum class ErrorSeverity {
    Recoverable,
    Transient,
    Permanent
};

enum class ErrorCategory {
    Timeout,
    NetworkError,
    HardwareError,
    InvalidArgument,
    InternalError,
    Unknown
};

struct CommError {
    ErrorCategory category;
    ErrorSeverity severity;
    std::string message;
    int device_id;
    cudaStream_t stream;
    std::chrono::steady_clock::time_point timestamp;

    bool is_recoverable() const {
        return severity == ErrorSeverity::Recoverable ||
               severity == ErrorSeverity::Transient;
    }
};

class HealthMonitor {
public:
    static HealthMonitor& instance();

    void start();
    void stop();
    bool is_running() const;

    void register_comm(void* comm_id, cudaStream_t stream);
    void unregister_comm(void* comm_id);

    void set_timeout_threshold(std::chrono::seconds timeout);
    void set_check_interval(std::chrono::milliseconds interval);
    void set_stall_confirmations(int count);

    using ErrorCallback = std::function<void(const CommError&)>;
    void set_error_callback(ErrorCallback callback);

    struct HealthStatus {
        void* comm_id;
        bool is_healthy;
        bool is_stalled;
        std::chrono::milliseconds last_progress;
    };

    std::optional<HealthStatus> get_status(void* comm_id) const;

private:
    HealthMonitor() = default;
    ~HealthMonitor();
    HealthMonitor(const HealthMonitor&) = delete;
    HealthMonitor& operator=(const HealthMonitor&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RetryHandler {
public:
    static RetryHandler& instance();

    struct RetryConfig {
        int max_attempts = 3;
        std::chrono::milliseconds initial_delay{100};
        std::chrono::milliseconds max_delay{10000};
        double backoff_multiplier = 2.0;
        bool use_jitter = true;
    };

    void set_config(const RetryConfig& config);
    RetryConfig get_config() const;

    template<typename Func>
    auto execute_with_retry(Func&& func, const std::string& operation_name)
        -> std::invoke_result_t<Func>;

    void reset_circuit_breaker();
    bool is_circuit_open() const;

private:
    RetryHandler() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ErrorClassifier {
public:
    static ErrorClassifier& instance();

    ErrorCategory classify_from_nccl(int nccl_result);
    ErrorSeverity classify_severity(ErrorCategory category, const std::string& details);

    bool is_transient_error(ErrorCategory category) const;
    bool is_permanent_error(ErrorCategory category) const;

    std::string get_error_message(ErrorCategory category) const;

private:
    ErrorClassifier() = default;

    bool is_timeout_indicative(ErrorCategory category) const;
    bool is_network_indicative(ErrorCategory category) const;
};

class CommErrorRecovery {
public:
    static CommErrorRecovery& instance();

    void initialize(HealthMonitor::ErrorCallback error_callback);
    void shutdown();

    void on_comm_error(const CommError& error);
    bool should_retry(const CommError& error);

    void recreate_communicator(int device_id);

    template<typename CollectiveFunc>
    auto wrap_collective(CollectiveFunc&& func,
                         int device_id,
                         cudaStream_t stream,
                         const std::string& operation_name)
        -> std::invoke_result_t<CollectiveFunc>;

    void set_timeout_threshold(std::chrono::seconds timeout);

private:
    CommErrorRecovery() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nova::comm
