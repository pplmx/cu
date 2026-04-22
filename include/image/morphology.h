#pragma once

#include <cstddef>
#include <cstdint>

void sharpenImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  float strength = 1.0f);

void applyThreshold(const uint8_t* d_input, uint8_t* d_output,
                    size_t width, size_t height,
                    uint8_t threshold);

void erodeImage(const uint8_t* d_input, uint8_t* d_output,
                size_t width, size_t height,
                int kernel_size = 3);

void dilateImage(const uint8_t* d_input, uint8_t* d_output,
                 size_t width, size_t height,
                 int kernel_size = 3);

void openingImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  int kernel_size = 3);

void closingImage(const uint8_t* d_input, uint8_t* d_output,
                  size_t width, size_t height,
                  int kernel_size = 3);
