# CUDA
FIND_PACKAGE(CUDA REQUIRED)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;--expt-relaxed-constexpr;-Xptxas; -v")
SET(CUDA_PROPAGATE_HOST_FLAGS FALSE)
SET(CUDA_SEPARABLE_COMPILATION ON)


# Qt
FIND_PACKAGE(Qt5OpenGL REQUIRED)
SET(CMAKE_AUTOMOC ON)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)


# ExTh
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${GpuMesh_SRC_DIR}/../ExperimentalTheatre/")
FIND_PACKAGE(ExperimentalTheatre REQUIRED)


# Global
SET(GpuMesh_LIBRARIES
    ${ExTh_LIBRARIES}
    pthread
    cgns)
SET(GpuMesh_INCLUDE_DIRS
    ${GpuMesh_SRC_DIR}
    ${ExTh_INCLUDE_DIRS})
SET(GpuMesh_QT_MODULES
    OpenGL)
