## Headers ##

# All the header files #
SET(GpuMesh_HEADERS
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.h
    ${GpuMesh_SRC_DIR}/Mesh.h)



## Sources ##

# All the source files #
SET(GpuMesh_SOURCES
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.cpp
    ${GpuMesh_SRC_DIR}/Mesh.cpp
    ${GpuMesh_SRC_DIR}/main.cpp)


## Resrouces ##

# Shaders
SET(GpuMesh_SHADERS
    ${GpuMesh_SRC_DIR}/resources/shaders/LitMesh.vert
    ${GpuMesh_SRC_DIR}/resources/shaders/LitMesh.frag
    ${GpuMesh_SRC_DIR}/resources/shaders/UnlitMesh.vert
    ${GpuMesh_SRC_DIR}/resources/shaders/UnlitMesh.frag
    ${GpuMesh_SRC_DIR}/resources/shaders/Shadow.vert
    ${GpuMesh_SRC_DIR}/resources/shaders/Shadow.frag
    ${GpuMesh_SRC_DIR}/resources/shaders/Backdrop.vert
    ${GpuMesh_SRC_DIR}/resources/shaders/Backdrop.frag)

# Qrc File
QT5_ADD_RESOURCES(GpuMesh_RESOURCES
    ${GpuMesh_SRC_DIR}/resources/GpuMesh.qrc)


## Global ##
SET(GpuMesh_CONFIG_FILES
    ${GpuMesh_SRC_DIR}/CMakeLists.txt
    ${GpuMesh_SRC_DIR}/FileLists.cmake
    ${GpuMesh_SRC_DIR}/LibLists.cmake)
	
SET(GpuMesh_SRC_FILES
    ${GpuMesh_HEADERS}
    ${GpuMesh_SOURCES}
    ${GpuMesh_SHADERS}
    ${GpuMesh_RESOURCES}
    ${GpuMesh_CONFIG_FILES}
    ${GpuMesh_MOC_CPP_FILES})
