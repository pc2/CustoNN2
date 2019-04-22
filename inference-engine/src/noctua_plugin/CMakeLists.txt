# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


set (TARGET_NAME "NoctuaPlugin")

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

include_directories(
    ${IE_MAIN_SOURCE_DIR}/src/inference_engine
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# OPENCL Libraries
include_directories($(shell aocl compile-config))
link_directories($(shell aocl compile-config))


add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} inference_engine ${INTEL_ITT_LIBS}  format_reader gflags)
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
