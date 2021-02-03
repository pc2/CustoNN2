# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8)

set (TARGET_NAME "test_plugin")
find_package(MPI)
file (GLOB SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SRC})

include_directories(
    ${IE_MAIN_SOURCE_DIR}/src/noctua_plugin
    /opt/intelFPGA_pro/19.1/hld/host/include
    )


link_directories(/opt/intelFPGA_pro/19.1/hld/host/linux64/lib /opt/intelFPGA_pro/19.1/hld/board/bittware_pcie/s10/linux64/lib)


include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})


link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${SRC})

#set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
#COMPILE_PDB_NAME ${TARGET_NAME})

if(MPI_COMPILE_FLAGS)
  set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(${TARGET_NAME} PROPERTIES
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()


target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} IE::ie_cpu_extension format_reader gflags OpenCL alteracl bitt_s10_pcie_mmd elf)
target_link_libraries(${TARGET_NAME} ${MPI_CXX_LIBRARIES})
if(UNIX)
    target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()
