/**
 * @file logger.hpp
 * @brief Structured logging infrastructure
 * @defgroup logger Structured Logging
 * @ingroup observability
 *
 * Provides structured logging with multiple severity levels:
 * - ERROR: Failures preventing operation
 * - WARN: Recoverable issues, degraded performance
 * - INFO: Significant milestones
 * - DEBUG: Detailed trace information
 * - TRACE: Per-iteration logging (disabled by default)
 *
 * Example usage:
 * @code
 * NOVA_LOG_INFO("operation=allocate", "bytes=" << size);
 * NOVA_LOG_WARN("device=" << device_id, "reason=low_memory");
 * @endcode
 *
 * Compile-time filtering via NOVA_LOG_LEVEL
 */

#ifndef NOVA_CUDA_LOGGER_HPP
#define NOVA_CUDA_LOGGER_HPP

#include <cstdio>
#include <string>
#include <sstream>
#include <chrono>

namespace nova {
namespace logging {

/**
 * @brief Log severity levels
 * @enum LogLevel
 * @ingroup logger
 */
enum class LogLevel {
    ERROR = 0,
    WARN = 1,
    INFO = 2,
    DEBUG = 3,
    TRACE = 4
};

/**
 * @brief Get current log level
 * @return Active log level
 * @note Can be overridden at compile time via NOVA_LOG_LEVEL
 */
inline LogLevel get_log_level() {
#if defined(NOVA_LOG_LEVEL_TRACE)
    return LogLevel::TRACE;
#elif defined(NOVA_LOG_LEVEL_DEBUG)
    return LogLevel::DEBUG;
#elif defined(NOVA_LOG_LEVEL_INFO)
    return LogLevel::INFO;
#elif defined(NOVA_LOG_LEVEL_WARN)
    return LogLevel::WARN;
#elif defined(NOVA_LOG_LEVEL_ERROR)
    return LogLevel::ERROR;
#else
    return LogLevel::INFO;
#endif
}

/**
 * @brief Check if a level should be logged
 * @param level Level to check
 * @return true if level should be output
 */
inline bool should_log(LogLevel level) {
    return static_cast<int>(level) <= static_cast<int>(get_log_level());
}

/**
 * @brief Format timestamp for log output
 * @return ISO 8601 formatted timestamp
 */
inline std::string format_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&time));
    return std::string(buf) + "." + std::to_string(ms.count());
}

/**
 * @brief Log a message at specified level
 * @param level Log level
 * @param context Key=value context pairs
 * @param message Log message
 */
inline void log(LogLevel level, const char* context, const char* message) {
    if (!should_log(level)) return;

    const char* level_str;
    switch (level) {
        case LogLevel::ERROR: level_str = "ERROR"; break;
        case LogLevel::WARN:  level_str = "WARN";  break;
        case LogLevel::INFO:  level_str = "INFO";  break;
        case LogLevel::DEBUG: level_str = "DEBUG"; break;
        case LogLevel::TRACE: level_str = "TRACE"; break;
    }

    std::fprintf(stderr, "[%s] [%s] [%s] %s\n",
                 format_timestamp().c_str(),
                 level_str,
                 context ? context : "",
                 message);
}

}  // namespace logging
}  // namespace nova

/**
 * @def NOVA_LOG_ERROR(context, message)
 * Log ERROR level message
 * @ingroup logger
 */
#define NOVA_LOG_ERROR(context, message) \
    nova::logging::log(nova::logging::LogLevel::ERROR, context, message)

/**
 * @def NOVA_LOG_WARN(context, message)
 * Log WARN level message
 * @ingroup logger
 */
#define NOVA_LOG_WARN(context, message) \
    nova::logging::log(nova::logging::LogLevel::WARN, context, message)

/**
 * @def NOVA_LOG_INFO(context, message)
 * Log INFO level message
 * @ingroup logger
 */
#define NOVA_LOG_INFO(context, message) \
    nova::logging::log(nova::logging::LogLevel::INFO, context, message)

/**
 * @def NOVA_LOG_DEBUG(context, message)
 * Log DEBUG level message
 * @ingroup logger
 */
#define NOVA_LOG_DEBUG(context, message) \
    nova::logging::log(nova::logging::LogLevel::DEBUG, context, message)

/**
 * @def NOVA_LOG_TRACE(context, message)
 * Log TRACE level message
 * @ingroup logger
 */
#define NOVA_LOG_TRACE(context, message) \
    nova::logging::log(nova::logging::LogLevel::TRACE, context, message)

#endif  // NOVA_CUDA_LOGGER_HPP
