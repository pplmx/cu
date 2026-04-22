# Stage 3: Parallel Algorithms - Design Spec

## Overview

- **Project**: CUDA 并行算法实现
- **目标**: 实现三个经典并行算法(Reduce, Scan, Sort),从基础版到优化版学习并行计算核心技巧
- **产出**: 6个算法实现(每算法基础+优化) + GoogleTest 测试

## Architecture

```
include/
  reduce.h       # 归约算法
  scan.h         # 前缀和
  sort.h         # 排序算法

src/
  reduce.cu      # 实现+优化版本
  scan.cu
  sort.cu

tests/
  reduce_test.cu
  scan_test.cu
  sort_test.cu
```

## Algorithms

### 1. Reduce (归约)

**问题**: 将n个元素归约为一个值(求和/最大/最小)

**基础版**:
- 相邻元素配对,每步减少一半
- log₂(n) 步完成
- 每个thread处理一对元素

**优化版**:
- **Warp级Reduce**: 同一warp内thread无需sync
- **Loop Unrolling**: 减少循环开销
- **多个block**: 扩展到大规模数据
- **atomicAdd参考**: 验证正确性

**参数**: 操作类型 (sum/max/min)

### 2. Scan (前缀和)

**问题**: 计算前缀和,output[i] = Σinput[0..i-1]

**基础版 (Kogge-Stone)**:
- 宽度每步翻倍: 1→2→4→8
- 算法清晰,易于理解
- O(n log n) 工作量

**优化版 (Blelloch)**:
- 上扫(构建树) + 下扫(传递)
- O(n log n) 但常数更小
- Shared memory加速
- 工作量更少

**参数**: 是否排他扫描

### 3. Sort (排序)

**基础版 (Odd-Even Transposition Sort)**:
- 奇偶相位交替交换
- 需要n次迭代
- O(n²) 复杂度
- 简单但低效

**优化版 (Bitonic Sort)**:
- 比较网络排序
- Batcher's算法
- O(log² n) 复杂度
- 更高效的并行排序

**参数**: 升序/降序

## Modern CUDA Features

- **Template**: 支持 int/float/double 不同类型
- **constexpr**: 编译期常量
- **Lambda**: 设备端操作函数
- **if constexpr**: 编译期分支
- **Device函数**: `__device__` 标记

## Testing Strategy

### Reduce Tests

| 测试 | 内容 |
|------|------|
| Sum测试 | [1..N] 求和 = N*(N+1)/2 |
| Max测试 | 找最大值和位置 |
| 边界测试 | 空数组,单元素 |
| 优化版一致性 | 优化版结果与基础版一致 |

### Scan Tests

| 测试 | 内容 |
|------|------|
| 前缀和 | 验证 output[i] = Σinput[0..i-1] |
| 边界 | 空数组,单元素 |
| 长度 | 2的幂次方,非2的幂次方 |
| 排他扫描 | exclusive prefix sum |

### Sort Tests

| 测试 | 内容 |
|------|------|
| 随机数组 | 验证排序正确性 |
| 重复元素 | 有相同值的数组 |
| 已排序 | 边界情况 |
| 反序 | 边界情况 |
| 长度 | 2的幂次方,非2的幂次方 |

## Acceptance Criteria

- [ ] Reduce: 基础版和优化版都正确
- [ ] Reduce: 优化版有加速效果(与atomicAdd对比)
- [ ] Scan: 正确的前缀和
- [ ] Scan: 边界处理正确
- [ ] Sort: Bitonic Sort正确排序任意长度数组
- [ ] 所有算法有benchmark性能测试
- [ ] GoogleTest测试全部通过
