# External Integrations

**Mapped:** 2026-04-23

## No External Service Integrations

This is a pure compute library with no external service dependencies:

- **No database connections**
- **No REST/gRPC APIs**
- **No authentication providers**
- **No cloud services**

## Hardware Requirements

| Requirement | Specification |
|-------------|---------------|
| GPU | CUDA-capable GPU required |
| Compute Capability | 6.0+ (Pascal and newer) |
| VRAM | Varies by workload |

## Docker Integration

The project includes Docker support for reproducible builds:

- **Base image**: CUDA-enabled container
- **Use case**: Build environment without local CUDA toolkit

## Version Control

| Service | Usage |
|---------|-------|
| **GitHub** | Source hosting, CI/CD |
| **Coveralls** | Code coverage tracking |

## CI/CD Integrations

GitHub Actions workflows:
- `ci.yml` - Main CI pipeline
- `docker.yml` - Docker image builds
- `cd.yml` - Continuous deployment
- `release.yml` - Release automation
