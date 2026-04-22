#pragma once

#include "cuda/memory/buffer.h"

namespace cuda::algo {

void matrixAdd(memory::Buffer<float> a,
               memory::Buffer<float> b,
               memory::Buffer<float> c,
               int numRows, int numCols);

}
