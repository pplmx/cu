#pragma once

#include <csignal>
#include <functional>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace nova::preemption {

enum class ShutdownPhase {
    Idle,
    Signaling,
    Checkpointing,
    Finalizing,
    Complete
};

struct ShutdownConfig {
    std::chrono::seconds shutdown_timeout{30};
    bool checkpoint_on_shutdown{true};
    bool validate_checkpoint_before_save{true};
    int max_checkpoint_retries{3};
    bool coordinated_checkpoint{true};
};

class SignalHandler {
public:
    static SignalHandler& instance();

    void install_handlers();
    void uninstall_handlers();

    bool is_shutdown_requested() const;
    int received_signal() const;

    using ShutdownCallback = std::function<void(int signal)>;
    void set_shutdown_callback(ShutdownCallback callback);

    struct HandlerState {
        bool handler_installed;
        bool shutdown_requested;
        int received_signal_number;
        std::chrono::steady_clock::time_point signal_received_at;
    };

    HandlerState get_state() const;

private:
    SignalHandler() = default;
    ~SignalHandler() = default;

    static void signal_handler(int signal);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ShutdownCoordinator {
public:
    static ShutdownCoordinator& instance();

    void initialize(const ShutdownConfig& config);
    void shutdown();

    void request_shutdown(int signal);

    ShutdownPhase get_phase() const;
    bool is_shutdown_in_progress() const;
    bool is_shutdown_complete() const;

    void begin_graceful_shutdown();
    void checkpoint_coordinated();
    void finalize_shutdown();

    using ShutdownStageCallback = std::function<void(ShutdownPhase)>;
    void set_stage_callback(ShutdownStageCallback callback);

    std::chrono::milliseconds get_elapsed_time() const;
    std::chrono::seconds get_remaining_timeout() const;

    bool extend_timeout(std::chrono::seconds additional_time);

private:
    ShutdownCoordinator() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ResumeValidator {
public:
    static ResumeValidator& instance();

    struct ValidationResult {
        bool is_valid;
        bool has_model_state;
        bool has_optimizer_state;
        bool has_rng_state;
        int checkpoint_step;
        std::string error_message;
        std::vector<std::string> warnings;
    };

    ValidationResult validate_checkpoint(const std::string& checkpoint_path);

    bool recover_state(const std::string& checkpoint_path);

    struct RecoveryResult {
        bool success;
        int recovered_step;
        std::string error_message;
    };

    RecoveryResult attempt_recovery(const std::string& checkpoint_path);

    std::string get_latest_checkpoint_path() const;

    void set_checkpoint_dir(const std::string& dir);

private:
    ResumeValidator() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class PreemptionManager {
public:
    static PreemptionManager& instance();

    void initialize(const ShutdownConfig& config);
    void shutdown();

    void on_preemption_signal(int signal);

    bool is_shutdown_requested() const;
    void wait_for_shutdown();

    bool request_timeout_extension(std::chrono::seconds additional_time);

    using PreemptionCallback = std::function<void(int signal)>;
    void set_preemption_callback(PreemptionCallback callback);

    struct Status {
        bool preemption_handlers_installed;
        bool shutdown_in_progress;
        bool shutdown_complete;
        int received_signal;
        std::chrono::milliseconds shutdown_elapsed;
        std::chrono::seconds remaining_timeout;
    };

    Status get_status() const;

private:
    PreemptionManager() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nova::preemption
