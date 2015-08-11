## Headers ##

# All the header files #
SET(GpuMesh_DATASTRUCTURES_HEADERS
    ${GpuMesh_SRC_DIR}/DataStructures/Mesh.h
    ${GpuMesh_SRC_DIR}/DataStructures/GpuMesh.h
    ${GpuMesh_SRC_DIR}/DataStructures/OptionMap.h
    ${GpuMesh_SRC_DIR}/DataStructures/OptimizationPlot.h
    ${GpuMesh_SRC_DIR}/DataStructures/Tetrahedron.h
    ${GpuMesh_SRC_DIR}/DataStructures/TetList.h
    ${GpuMesh_SRC_DIR}/DataStructures/TetPool.h
    ${GpuMesh_SRC_DIR}/DataStructures/Triangle.h
    ${GpuMesh_SRC_DIR}/DataStructures/TriSet.h)

SET(GpuMesh_EVALUATORS_HEADERS
    ${GpuMesh_SRC_DIR}/Evaluators/AbstractEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/InsphereEdgeEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/MeanRatioEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/SolidAngleEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/VolumeEdgeEvaluator.cpp)

SET(GpuMesh_MESHERS_HEADERS
    ${GpuMesh_SRC_DIR}/Meshers/AbstractMesher.h
    ${GpuMesh_SRC_DIR}/Meshers/CpuDelaunayMesher.h
    ${GpuMesh_SRC_DIR}/Meshers/CpuParametricMesher.h)

SET(GpuMesh_RENDERERS_HEADERS
    ${GpuMesh_SRC_DIR}/Renderers/AbstractRenderer.h
    ${GpuMesh_SRC_DIR}/Renderers/ScaffoldRenderer.h
    ${GpuMesh_SRC_DIR}/Renderers/SurfacicRenderer.h)

SET(GpuMesh_SMOOTHERS_HEADERS
    ${GpuMesh_SRC_DIR}/Smoothers/SmoothingHelper.h
    ${GpuMesh_SRC_DIR}/Smoothers/AbstractSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/SpringLaplaceSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/QualityLaplaceSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/LocalOptimisationSmoother.h)

SET(GpuMesh_UITABS_HEADERS
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/MeshTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/EvaluateTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/SmoothTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/RenderTab.h)

SET(GpuMesh_USERINTERFACE_HEADERS
    ${GpuMesh_UITABS_HEADERS}
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.h)

SET(GpuMesh_HEADERS
    ${GpuMesh_DATASTRUCTURES_HEADERS}
    ${GpuMesh_EVALUATORS_HEADERS}
    ${GpuMesh_MESHERS_HEADERS}
    ${GpuMesh_RENDERERS_HEADERS}
    ${GpuMesh_SMOOTHERS_HEADERS}
    ${GpuMesh_USERINTERFACE_HEADERS}
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.h)


## Sources ##

# All the source files #
SET(GpuMesh_DATASTRUCTURES_SOURCES
    ${GpuMesh_SRC_DIR}/DataStructures/Mesh.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/GpuMesh.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/OptimizationPlot.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TetList.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TetPool.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TriSet.cpp)

SET(GpuMesh_EVALUATORS_SOURCES
    ${GpuMesh_SRC_DIR}/Evaluators/AbstractEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/InsphereEdgeEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/MeanRatioEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/SolidAngleEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/VolumeEdgeEvaluator.cpp)

SET(GpuMesh_MESHERS_SOURCES
    ${GpuMesh_SRC_DIR}/Meshers/AbstractMesher.cpp
    ${GpuMesh_SRC_DIR}/Meshers/CpuDelaunayMesher.cpp
    ${GpuMesh_SRC_DIR}/Meshers/CpuParametricMesher.cpp)

SET(GpuMesh_RENDERERS_SOURCES
    ${GpuMesh_SRC_DIR}/Renderers/AbstractRenderer.cpp
    ${GpuMesh_SRC_DIR}/Renderers/ScaffoldRenderer.cpp
    ${GpuMesh_SRC_DIR}/Renderers/SurfacicRenderer.cpp)

SET(GpuMesh_SMOOTHERS_SOURCES
    ${GpuMesh_SRC_DIR}/Smoothers/SmoothingHelper.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/AbstractSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/SpringLaplaceSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/QualityLaplaceSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/LocalOptimisationSmoother.cpp)

SET(GpuMesh_UITABS_SOURCES
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/MeshTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/EvaluateTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/SmoothTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/RenderTab.cpp)

SET(GpuMesh_USERINTERFACE_SOURCES
    ${GpuMesh_UITABS_SOURCES}
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.cpp)

SET(GpuMesh_SOURCES
    ${GpuMesh_DATASTRUCTURES_SOURCES}
    ${GpuMesh_EVALUATORS_SOURCES}
    ${GpuMesh_MESHERS_SOURCES}
    ${GpuMesh_RENDERERS_SOURCES}
    ${GpuMesh_SMOOTHERS_SOURCES}
    ${GpuMesh_USERINTERFACE_SOURCES}
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.cpp
    ${GpuMesh_SRC_DIR}/main.cpp)


## UI
SET(GpuMesh_UI_FILES
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.ui)
QT5_WRAP_UI(GpuMesh_UI_SRCS ${GpuMesh_UI_FILES})


## Resrouces ##
SET(GpuMesh_SHADER_DIR
    ${GpuMesh_SRC_DIR}/resources/shaders)

# Genereic shaders
SET(GpuMesh_GENERIC_SHADERS
    ${GpuMesh_SHADER_DIR}/generic/QualityLut.glsl
    ${GpuMesh_SHADER_DIR}/generic/Lighting.glsl)

# Vertex shaders
SET(GpuMesh_VERTEX_SHADERS
    ${GpuMesh_SHADER_DIR}/vertex/Shadow.vert
    ${GpuMesh_SHADER_DIR}/vertex/LitMesh.vert
    ${GpuMesh_SHADER_DIR}/vertex/UnlitMesh.vert
    ${GpuMesh_SHADER_DIR}/vertex/ScaffoldJoint.vert
    ${GpuMesh_SHADER_DIR}/vertex/ScaffoldTube.vert
    ${GpuMesh_SHADER_DIR}/vertex/Wireframe.vert
    ${GpuMesh_SHADER_DIR}/vertex/Bloom.vert
    ${GpuMesh_SHADER_DIR}/vertex/Filter.vert)

# Fragment shaders
SET(GpuMesh_FRAGMENT_SHADERS
    ${GpuMesh_SHADER_DIR}/fragment/Shadow.frag
    ${GpuMesh_SHADER_DIR}/fragment/LitMesh.frag
    ${GpuMesh_SHADER_DIR}/fragment/UnlitMesh.frag
    ${GpuMesh_SHADER_DIR}/fragment/ScaffoldJoint.frag
    ${GpuMesh_SHADER_DIR}/fragment/ScaffoldTube.frag
    ${GpuMesh_SHADER_DIR}/fragment/Wireframe.frag
    ${GpuMesh_SHADER_DIR}/fragment/BloomBlur.frag
    ${GpuMesh_SHADER_DIR}/fragment/BloomBlend.frag
    ${GpuMesh_SHADER_DIR}/fragment/Gradient.frag
    ${GpuMesh_SHADER_DIR}/fragment/Screen.frag
    ${GpuMesh_SHADER_DIR}/fragment/Brush.frag
    ${GpuMesh_SHADER_DIR}/fragment/Grain.frag)

# Compute shaders
SET(GpuMesh_BOUNDARY_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Boundary/None.glsl
    ${GpuMesh_SHADER_DIR}/compute/Boundary/Box.glsl
    ${GpuMesh_SHADER_DIR}/compute/Boundary/Sphere.glsl
    ${GpuMesh_SHADER_DIR}/compute/Boundary/ElbowPipe.glsl)

SET(GpuMesh_MEASURING_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Measuring/TetrahedraEvaluation.glsl
    ${GpuMesh_SHADER_DIR}/compute/Measuring/PrismsEvaluation.glsl
    ${GpuMesh_SHADER_DIR}/compute/Measuring/HexahedraEvaluation.glsl
    ${GpuMesh_SHADER_DIR}/compute/Measuring/SimultaneousEvaluation.glsl
    ${GpuMesh_SHADER_DIR}/compute/Measuring/StatisticsReduction.glsl)

SET(GpuMesh_QUALITY_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Quality/QualityInterface.glsl
    ${GpuMesh_SHADER_DIR}/compute/Quality/InsphereEdge.glsl
    ${GpuMesh_SHADER_DIR}/compute/Quality/MeanRatio.glsl
    ${GpuMesh_SHADER_DIR}/compute/Quality/SolidAngle.glsl
    ${GpuMesh_SHADER_DIR}/compute/Quality/VolumeEdge.glsl)

SET(GpuMesh_SMOOTHING_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/SmoothingHelper.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/SpringLaplace.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/QualityLaplace.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/LocalOptimisation.glsl)


SET(GpuMesh_COMPUTE_SHADERS
    ${GpuMesh_BOUNDARY_SHADERS}
    ${GpuMesh_MEASURING_SHADERS}
    ${GpuMesh_QUALITY_SHADERS}
    ${GpuMesh_SMOOTHING_SHADERS}
    ${GpuMesh_SHADER_DIR}/compute/Mesh.glsl)

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
    ${GpuMesh_UI_SRCS}
    ${GpuMesh_GENERIC_SHADERS}
    ${GpuMesh_VERTEX_SHADERS}
    ${GpuMesh_FRAGMENT_SHADERS}
    ${GpuMesh_COMPUTE_SHADERS}
    ${GpuMesh_RESOURCES}
    ${GpuMesh_CONFIG_FILES}
    ${GpuMesh_MOC_CPP_FILES})
