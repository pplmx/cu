add_executable(nvblox_metrics_test
    nvblox_metrics_test.cpp
)

target_link_libraries(nvblox_metrics_test
    PRIVATE
    GTest::gtest_main
    nova_core
    Threads::Threads
)

target_include_directories(nvblox_metrics_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

if(CUDA_FOUND)
    target_compile_definitions(nvblox_metrics_test PRIVATE CUDA_FOUND)
endif()

gtest_discover_tests(nvblox_metrics_test)
