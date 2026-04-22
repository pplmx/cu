cmake_minimum_required(VERSION 3.25)

project(cuda-samples
        VERSION 0.0.1
        DESCRIPTION "A CUDA parallel algorithms library"
        HOMEPAGE_URL "https://github.com/pplmx/cuda-samples"
        LANGUAGES CXX CUDA
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

set(CUDA_KERNEL_DIR ${CMAKE_SOURCE_DIR}/include/cuda/kernel)
set(CUDA_ALGO_DIR ${CMAKE_SOURCE_DIR}/include/cuda/algo)
set(CUDA_API_DIR ${CMAKE_SOURCE_DIR}/include/cuda/api)
set(CUDA_SRC_KERNEL ${CMAKE_SOURCE_DIR}/src/kernel)
set(CUDA_SRC_ALGO ${CMAKE_SOURCE_DIR}/src/algo)

add_library(cuda_kernel INTERFACE)
target_include_directories(cuda_kernel INTERFACE
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
        $<BUILD_INTERFACE:${CUDA_SRC_KERNEL}>
        $<BUILD_INTERFACE:${CUDA_SRC_ALGO}>
)
target_link_libraries(cuda_kernel INTERFACE CUDA::cudart)
set_target_properties(cuda_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_library(cuda_algo INTERFACE)
target_include_directories(cuda_algo INTERFACE
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
        $<BUILD_INTERFACE:${CUDA_KERNEL_DIR}>
)
target_link_libraries(cuda_algo INTERFACE
        cuda_kernel
        CUDA::cublas
)

add_library(cuda_api INTERFACE)
target_include_directories(cuda_api INTERFACE
        $<BUILD_INTERFACE:${CUDA_API_DIR}>
        $<BUILD_INTERFACE:${CUDA_ALGO_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CUDA_SRC_KERNEL}>
        $<BUILD_INTERFACE:${CUDA_SRC_ALGO}>
)
target_link_libraries(cuda_api INTERFACE cuda_algo)

file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/*.cu
)

add_library(cuda_impl STATIC ${CUDA_SOURCES})
target_include_directories(cuda_impl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
target_link_libraries(cuda_impl PUBLIC
        CUDA::cudart
        CUDA::cublas
)
set_target_properties(cuda_impl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_impl cuda_api)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/data)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
