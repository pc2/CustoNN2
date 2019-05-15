# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXCompilerFlag)

if (ENABLE_SANITIZER)
    set(SANITIZER_COMPILER_FLAGS "-fsanitize=address")
    CHECK_CXX_COMPILER_FLAG("-fsanitize-recover=address" SANITIZE_RECOVER_SUPPORTED)
    if (SANITIZE_RECOVER_SUPPORTED)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=address")
    endif()
    set(SANITIZER_LINKER_FLAGS "-fsanitize=address -fuse-ld=gold")

    set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
endif()
