# Phase 36: Build Performance

**Milestone:** v1.8 Developer Experience
**Status:** Planning

## Goal

Developers can build Nova quickly using CMake presets and ccache

## Requirements

| ID | Description |
|----|-------------|
| BLD-01 | CMakePresets.json with dev/release/ci presets |
| BLD-02 | NOVA_USE_CCACHE CMake option for ccache |
| BLD-03 | NOVA_ENABLE_UNITY_BUILD CMake option |
| BLD-04 | Build performance documentation |

## Success Criteria

1. Developer can configure and build using `cmake --preset dev` / `cmake --build --preset dev`
2. Developer can enable ccache with `-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`
3. Developer can enable unity builds with `-DNOVA_ENABLE_UNITY_BUILD=ON`
4. Developer can follow build performance documentation

## Implementation

### 1. CMakePresets.json

```json
{
  "version": 10,
  "configurePresets": [
    {
      "name": "dev",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "NOVA_USE_CCACHE": "ON"
      }
    },
    {
      "name": "release",
      "binaryDir": "${sourceDir}/build-release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release"
      }
    },
    {
      "name": "ci",
      "binaryDir": "${sourceDir}/build-ci",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "NOVA_ENABLE_UNITY_BUILD": "ON"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "dev",
      "configurePreset": "dev"
    },
    {
      "name": "release",
      "configurePreset": "release"
    },
    {
      "name": "ci",
      "configurePreset": "ci"
    }
  ]
}
```

### 2. CMake Options

Add to CMakeLists.txt:

- `NOVA_USE_CCACHE` - Enable ccache detection and configuration
- `NOVA_ENABLE_UNITY_BUILD` - Already exists, ensure it's properly documented

### 3. Documentation

Create `docs/build-performance.md` with:

- How to use CMake presets
- ccache setup and usage
- Unity build guidance
- Performance tips

---

## Context created: 2026-04-26
