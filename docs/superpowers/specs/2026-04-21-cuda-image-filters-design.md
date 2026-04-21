# Stage 2: CUDA Image Filters - Design Spec

## Overview

- **Project**: CUDA 图像滤波器实现
- **目标**: 实现三个常用图像滤波器,学习 CUDA 图像处理技巧,建立测试体系
- **产出**: 3个滤波器实现 + GoogleTest 测试

## Architecture

### 组件

```
include/
  image_utils.h        # ImageBuffer, 公共工具
  gaussian_blur.h      # 高斯模糊
  sobel_edge.h         # Sobel边缘检测
  brightness.h         # 亮度/对比度

src/
  image_utils.cu       # 图像处理kernel和工具函数
  gaussian_blur.cu
  sobel_edge.cu
  brightness.cu
  main.cpp             # 演示程序

tests/
  image_utils_test.cu  # 工具函数测试
  gaussian_blur_test.cu
  sobel_edge_test.cu
  brightness_test.cu

data/
  test_patterns.cuh    # 内置测试图案生成器
  stb_image.h          # 图像加载
```

## Filters

### 1. Gaussian Blur (高斯模糊)

**原理**: 可分离高斯卷积
- 2D 高斯核: G(x,y) = exp(-(x²+y²)/(2σ²))
- 可分离为两个 1D 卷积: 先水平,再垂直
- 计算量从 O(M×N×K²) 降到 O(2×M×N×K)

**CUDA 技巧**:
- Constant memory 存储 1D kernel
- 可分离卷积两步处理
- Tiled 共享内存优化(可选)

**参数**: sigma (默认 1.0), kernel_size (默认 3,5,7 奇数)

### 2. Sobel Edge Detection (边缘检测)

**原理**: 梯度检测
- Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] (水平梯度)
- Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] (垂直梯度)
- Magnitude = sqrt(Gx² + Gy²)

**CUDA 技巧**:
- 读取一次图像,计算 Gx 和 Gy
- In-place 或双buffer
- Threshold 可选

**输出**: 灰度边缘图或二值图(阈值可配)

### 3. Brightness/Contrast (亮度/对比度)

**原理**: 点操作
- output = alpha * input + beta
- alpha: 对比度 (默认 1.0)
- beta: 亮度偏移 (默认 0)

**CUDA 技巧**:
- 最简单的并行: 每个像素一个线程,完全无依赖
- 适合学习 CUDA 基本模式

**参数**: alpha (0.0-3.0), beta (-100, 100)

## Modern C++ Features

- **RAII**: ImageBuffer 用 unique_ptr 管理 GPU 内存
- **Template**: 支持 uchar3/float3 不同像素格式
- **Lambda**: 测试中生成预期值
- **std::array**: 固定大小数组代替 raw array
- **if constexpr**: 编译期分支
- **Concept**: C++20 concepts 约束模板参数(如果编译器支持)

## Testing Strategy

### 单元测试 (确定性,快速)

| 测试 | 内容 |
|------|------|
| ImageBuffer | 创建/销毁,数据传递 |
| Test Patterns | 棋盘格,渐变,单色 |
| 边界处理 | 小图像,奇数尺寸 |
| 内存管理 | RAII, 泄漏检测 |

### 集成测试 (真实数据)

| 测试 | 内容 |
|------|------|
| CPU vs GPU | 对比参考实现结果 |
| stb_image | 读取/处理/保存循环 |
| 性能基准 | 各滤波器执行时间 |

### 正确性标准

- CPU 和 GPU 结果误差 < 1e-5 (float)
- 边缘检测边界值在合理范围内

## Acceptance Criteria

- [ ] 三个滤波器编译无 warnings
- [ ] GoogleTest 单元测试全部通过
- [ ] CPU vs GPU 结果一致性验证通过
- [ ] 演示程序能处理真实图片

## Dependencies

- CUDA Toolkit (12.x)
- GoogleTest (submodule 或 FetchContent)
- stb_image.h (单头文件, MIT license)
