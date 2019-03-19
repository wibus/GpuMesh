# CUDA
FIND_PACKAGE(CUDA REQUIRED)
LIST(APPEND CUDA_NVCC_FLAGS "-Xptxas;-v;")
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;")
LIST(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr;")
SET(CUDA_PROPAGATE_HOST_FLAGS TRUE)
SET(CUDA_SEPARABLE_COMPILATION ON)


# Qt
FIND_PACKAGE(Qt5OpenGL REQUIRED)
SET(CMAKE_AUTOMOC ON)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)


# ExTh
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${GpuMesh_SRC_DIR}/../ExperimentalTheatre/")
FIND_PACKAGE(ExperimentalTheatre REQUIRED)

# Pirate
IF(DEFINED ENABLE_PIRATE)
	SET(Pirate_LIBRARIES
		optimized "${GpuMesh_SRC_DIR}/../common/apps/Linux_5.4.0/lib/libPir.a"
		debug "${GpuMesh_SRC_DIR}/../common/apps/dbgLinux_5.4.0/lib/libPir.a")
	SET(Pirate_INCLUDE_DIRS
		"${GpuMesh_SRC_DIR}/../common/apps/Linux_5.4.0/include")
ENDIF(DEFINED ENABLE_PIRATE)

# CGNS
IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CGNS_LIBRARIES "cgns")
    SET(CGNS_INCLUDE_DIRS)
ELSEIF(MSVC)
    SET(CGNS_LIBRARIES "C:/Program Files (x86)/cgns/lib/cgns.lib")
    SET(CGNS_DLLS "C:/Program Files (x86)/cgns/bin/cgnsdll.dll")
    SET(CGNS_INCLUDE_DIRS "C:/Program Files (x86)/cgns/include")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# PThread
IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(pthread_LIBRARIES "pthread")
ELSEIF(MSVC)
    SET(pthread_LIBRARIES)
ENDIF(CMAKE_COMPILER_IS_GNUCXX)


# Global
SET(GpuMesh_LIBRARIES
    ${ExTh_LIBRARIES}
    ${pthread_LIBRARIES}
	${CGNS_LIBRARIES}
	${Pirate_LIBRARIES})
SET(GpuMesh_DLLS
	${ExTh_DLLS}
	${CGNS_DLLS}
	${Pirate_DLLS})
SET(GpuMesh_INCLUDE_DIRS
    ${GpuMesh_SRC_DIR}
    ${ExTh_INCLUDE_DIRS}
	${CGNS_INCLUDE_DIRS}
	${Pirate_INCLUDE_DIRS})
SET(GpuMesh_QT_MODULES
    OpenGL)
