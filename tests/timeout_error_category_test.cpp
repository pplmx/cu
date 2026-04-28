#include <gtest/gtest.h>
#include <system_error>
#include "cuda/error/timeout.hpp"

using namespace nova::error;

class TimeoutErrorCategoryTest : public ::testing::Test {};

TEST_F(TimeoutErrorCategoryTest, NameReturnsTimeout) {
    timeout_error_category cat;
    EXPECT_STREQ(cat.name(), "timeout");
}

TEST_F(TimeoutErrorCategoryTest, MakeTimeoutErrorCreatesValidCode) {
    auto ec = make_timeout_error(timeout_error_code::operation_timed_out);
    EXPECT_EQ(ec.value(), static_cast<int>(timeout_error_code::operation_timed_out));
    EXPECT_STREQ(ec.category().name(), "timeout");
}

TEST_F(TimeoutErrorCategoryTest, MessageOperationTimedOut) {
    timeout_error_category cat;
    EXPECT_EQ(cat.message(static_cast<int>(timeout_error_code::operation_timed_out)), "Operation timed out");
}

TEST_F(TimeoutErrorCategoryTest, MessageDeadlineExceeded) {
    timeout_error_category cat;
    EXPECT_EQ(cat.message(static_cast<int>(timeout_error_code::deadline_exceeded)), "Deadline exceeded");
}

TEST_F(TimeoutErrorCategoryTest, MessageWatchdogTimeout) {
    timeout_error_category cat;
    EXPECT_EQ(cat.message(static_cast<int>(timeout_error_code::watchdog_timeout)), "Watchdog detected stalled operation");
}

TEST_F(TimeoutErrorCategoryTest, MessageTimeoutCancelled) {
    timeout_error_category cat;
    EXPECT_EQ(cat.message(static_cast<int>(timeout_error_code::timeout_cancelled)), "Timeout was cancelled");
}

TEST_F(TimeoutErrorCategoryTest, MessageUnknownForInvalidCode) {
    timeout_error_category cat;
    EXPECT_EQ(cat.message(999), "Unknown timeout error");
}

TEST_F(TimeoutErrorCategoryTest, RecoveryHintOperationTimedOut) {
    timeout_error_category cat;
    auto hint = cat.recovery_hint(static_cast<int>(timeout_error_code::operation_timed_out));
    EXPECT_TRUE(hint.find("Increase timeout") != std::string_view::npos);
}

TEST_F(TimeoutErrorCategoryTest, RecoveryHintDeadlineExceeded) {
    timeout_error_category cat;
    auto hint = cat.recovery_hint(static_cast<int>(timeout_error_code::deadline_exceeded));
    EXPECT_TRUE(hint.find("Parent operation") != std::string_view::npos);
}

TEST_F(TimeoutErrorCategoryTest, RecoveryHintWatchdogTimeout) {
    timeout_error_category cat;
    auto hint = cat.recovery_hint(static_cast<int>(timeout_error_code::watchdog_timeout));
    EXPECT_TRUE(hint.find("stalled") != std::string_view::npos);
}

TEST_F(TimeoutErrorCategoryTest, RecoveryHintTimeoutCancelled) {
    timeout_error_category cat;
    auto hint = cat.recovery_hint(static_cast<int>(timeout_error_code::timeout_cancelled));
    EXPECT_TRUE(hint.find("No action needed") != std::string_view::npos);
}

TEST_F(TimeoutErrorCategoryTest, TimeoutCategoryIsSingleton) {
    const auto& cat1 = timeout_category();
    const auto& cat2 = timeout_category();
    EXPECT_EQ(&cat1, &cat2);
}

TEST_F(TimeoutErrorCategoryTest, ErrorCodeEquivalence) {
    auto ec1 = make_timeout_error(timeout_error_code::operation_timed_out);
    auto ec2 = make_timeout_error(timeout_error_code::operation_timed_out);
    EXPECT_EQ(ec1, ec2);
}

TEST_F(TimeoutErrorCategoryTest, DefaultConstructorErrorCode) {
    std::error_code ec;
    EXPECT_FALSE(ec);
}
