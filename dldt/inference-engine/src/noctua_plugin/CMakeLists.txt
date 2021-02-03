# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


set (TARGET_NAME "NoctuaPlugin")
find_package(MPI REQUIRED)
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
    ${IE_MAIN_SOURCE_DIR}/samples/common/format_reader
    /opt/intelFPGA_pro/18.1.1/hld/host/include
)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
message(WARNING ${MPI_CXX_INCLUDE_PATH})
message(WARNING ${MPI_CXX_LIBRARIES})


#set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
#if(MPI_LINK_FLAGS)
 
#endif()

if(MPI_COMPILE_FLAGS)
  #set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
 set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()



# OPENCL Libraries
link_directories(/opt/intelFPGA_pro/18.1.1/hld/host/linux64/lib /opt/intelFPGA_pro/18.1.1/hld/board/nalla_pcie/linux64/lib)


add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} inference_engine ${INTEL_ITT_LIBS} ${OpenCV_LIBRARIES}  format_reader gflags OpenCL alteracl nalla_pcie_mmd elf)
target_link_libraries(${TARGET_NAME} ${MPI_CXX_LIBRARIES})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
