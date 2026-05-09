# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.14.0] - 2026-05-07

### Documentation

- Complete Doxygen coverage for all public headers (30 headers)
- Inline code comments across memory, device, and algo layers
- Structured logging infrastructure (ERROR/WARN/INFO/DEBUG/TRACE)
- User guides: SPARSE.md, QUANTIZATION.md, ARCHITECTURE.md
- Updated README with v2.x features

## [2.13.0] - 2026-05-06

### Added

- Transformer optimization features
- KV cache management improvements
- Flash attention support

## [2.12.0] - 2026-05-03

### Added

- Advanced quantization infrastructure
- FP8 support (E4M3, E5M2) for H100/H200
- Quantization-aware training (QAT) support
- Calibration framework (MinMax, Percentile)

## [2.11.0] - 2026-05-02

### Added

- Performance tooling
- Kernel profiling utilities
- Memory bandwidth analysis

## [2.10.0] - 2026-05-01

### Added

- Sparse solver acceleration
- CG, GMRES, BiCGSTAB iterative solvers
- Jacobi and ILU(0) preconditioners
- RCM bandwidth reduction reordering

## [2.9.0] - 2026-05-01

### Added

- Architecture refactoring
- Five-layer architecture finalization

## [2.8.0] - 2026-05-01

### Added

- Numerical computing enhancements
- Precision management

## [2.7.0] - 2026-04-30

### Added

- Comprehensive testing and validation
- Test coverage improvements

## [2.6.0] - 2026-04-29

### Added

- Transformer and inference optimization
- Optimized GEMM operations

## [2.5.0] - 2026-04-28

### Added

- Error handling and recovery
- Timeout management
- Circuit breaker patterns
- Retry logic with backoff

## [2.4.0] - 2026-04-28

### Added

- Production hardening
- Memory pool improvements
- Error reporting enhancements

## [2.3.0] - 2026-04-28

### Added

- Extended algorithm support
- Additional parallel primitives

## [2.2.0] - 2026-04-27

### Added

- Comprehensive enhancement
- Feature parity across modules

## [2.1.0] - 2026-04-26

### Added

- New algorithm implementations
- Performance optimizations

## [2.0.0] - 2026-04-26

### Added

- Testing and quality infrastructure
- Comprehensive test coverage

## [1.9.0] - 2026-04-26

### Added

- Documentation system
- API reference generation

## [1.8.0] - 2026-04-26

### Added

- Developer experience improvements
- CMake build enhancements

## [1.7.0] - 2026-04-26

### Added

- Benchmarking and testing
- Performance benchmarks

## [1.6.0] - 2026-04-26

### Added

- Performance and training optimizations
- Training-specific utilities

## [1.5.0] - 2026-04-26

### Added

- Fault tolerance
- Checkpoint and recovery

## [1.4.0] - 2026-04-24

### Added

- Multi-node support
    - MPI integration (MpiContext, rank discovery)
    - Topology detection (NIC enumeration, RDMA capability)
    - Cross-node communicators (MultiNodeContext, HierarchicalAllReduce)
- Build system improvements
    - Auto-detect CPU cores for parallel builds
    - Ninja generator support

## [1.3.0] - 2026-04-24

### Added

- NCCL integration
- Multi-GPU collectives

## [1.2.0] - 2026-04-24

### Added

- Toolchain upgrade
- Modern C++ support

## [1.1.0] - 2026-04-24

### Added

- Multi-GPU support
- Device memory pools

## [1.0.0] - 2026-04-24

### Added

- Five-layer architecture
    - Layer 0: `cuda::memory` - Buffer, unique_ptr, MemoryPool
    - Layer 1: `cuda::device` - Pure device kernels
    - Layer 2: `cuda::algo` - Algorithm wrappers
    - Layer 3: `cuda::api` - High-level API
- Core algorithms: reduce, scan, sort
- Image processing: brightness, gaussian_blur, sobel_edge
- Matrix operations: add, mult
- Convolution: 2D convolution
