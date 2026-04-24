# FindNCCL.cmake - Locate NCCL library and headers
#
# This module defines the following variables:
#   NCCL_FOUND          - True if the NCCL library and headers are found
#   NCCL_INCLUDE_DIRS   - The include directories for NCCL
#   NCCL_LIBRARIES      - The libraries to link against for NCCL
#   NCCL_VERSION        - The version string of NCCL (if available)
#
# Required version: NCCL 2.25 or higher
#
# Usage:
#   find_package(NCCL 2.25)  # Requires 2.25+
#   find_package(NCCL)       # Any version (not recommended)
#
# This module will search in the following locations:
#   1. Environment variable NCCL_DIR
#   2. /usr/local/nccl
#   3. /usr/local/cuda/nccl (CUDA Toolkit bundled NCCL)
#   4. CUDAToolkit include path
#   5. System default paths

# Early return if target is already defined
if(TARGET NCCL::nccl)
    return()
endif()

# Minimum NCCL version requirement (per STACK.md)
set(NCCL_MIN_VERSION "2.25")

# Define search paths in priority order
set(NCCL_SEARCH_PATHS
        ${NCCL_DIR}
        $ENV{NCCL_DIR}
        /usr/local/nccl
        /usr/local/cuda/nccl
        /opt/nccl
)

# Locate the NCCL header file (nccl.h)
find_path(NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${NCCL_DIR} $ENV{NCCL_DIR}
        PATHS ${NCCL_SEARCH_PATHS}
        PATH_SUFFIXES include include/nccl
        DOC "NCCL include directory containing nccl.h"
)

# Locate the NCCL library
# NCCL uses versioned sonames: libnccl.so.2, libnccl.so.2.x.x
find_library(NCCL_LIBRARY
        NAMES nccl
        HINTS ${NCCL_DIR} $ENV{NCCL_DIR}
        PATHS ${NCCL_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
        DOC "NCCL library (libnccl.so)"
)

# Set include directories and libraries
set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
set(NCCL_LIBRARIES ${NCCL_LIBRARY})

# Version detection from nccl.h
# NCCL encodes version as: MNNPP (e.g., 22500 = 2.25.0)
if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    # Try to extract version from nccl.h header
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_DEF
            REGEX "^#define[ \t]+NCCL_VERSION[ \t]+[0-9]+"
            LIMIT_COUNT 1
    )

    if(NCCL_VERSION_DEF)
        string(REGEX REPLACE "^#define[ \t]+NCCL_VERSION[ \t]+([0-9]+).*" "\\1"
                NCCL_VERSION_NUM "${NCCL_VERSION_DEF}")

        # Parse version number: MNNPP -> M.NN.PP
        math(EXPR NCCL_VERSION_MAJOR "${NCCL_VERSION_NUM} / 10000")
        math(EXPR NCCL_VERSION_MINOR "(${NCCL_VERSION_NUM} % 10000) / 100")
        math(EXPR NCCL_VERSION_PATCH "${NCCL_VERSION_NUM} % 100")

        set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    endif()
endif()

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
        REQUIRED_VARS
            NCCL_LIBRARY
            NCCL_INCLUDE_DIRS
        VERSION_VAR
            NCCL_VERSION
        FAIL_MESSAGE
            "NCCL not found. Set NCCL_DIR environment variable or install NCCL 2.25+. "
            "See https://developer.nvidia.com/nccl for installation instructions."
)

# Create imported target NCCL::nccl
if(NCCL_FOUND AND NOT TARGET NCCL::nccl)
    add_library(NCCL::nccl UNKNOWN IMPORTED)
    set_target_properties(NCCL::nccl PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
    )

    # Set version target property
    if(NCCL_VERSION)
        set_target_properties(NCCL::nccl PROPERTIES
                INTERFACE_NCCL_VERSION "${NCCL_VERSION}"
        )
    endif()
endif()

# Mark variables as advanced
mark_as_advanced(NCCL_INCLUDE_DIR)
mark_as_advanced(NCCL_LIBRARY)
