# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8)

set (TARGET_NAME "test_plugin")

file (GLOB SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SRC})

include_directories(
    ${IE_MAIN_SOURCE_DIR}/src/noctua_plugin
    /opt/intelFPGA_pro/18.1.1/hld/host/include
    )


link_directories(/opt/intelFPGA_pro/18.1.1/hld/host/linux64/lib /opt/intelFPGA_pro/18.1.1/hld/board/nalla_pcie/linux64/lib )

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${SRC})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} IE::ie_cpu_extension format_reader gflags OpenCL alteracl nalla_pcie_mmd elf)

if(UNIX)
    target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()
