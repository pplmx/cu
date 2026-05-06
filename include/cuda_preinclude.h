#pragma once

/**
 * @file cuda_preinclude.h
 * @brief Must be included BEFORE gtest and any system headers with extern "C"
 *
 * This header includes CUDA runtime first, before any extern "C" context.
 * It MUST be force-included at the start of every compilation unit using tests.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
