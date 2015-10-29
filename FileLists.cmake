## Headers ##

# All the header files #
SET(GpuMesh_DATASTRUCTURES_HEADERS
    ${GpuMesh_SRC_DIR}/DataStructures/Mesh.h
    ${GpuMesh_SRC_DIR}/DataStructures/GpuMesh.h
    ${GpuMesh_SRC_DIR}/DataStructures/MeshCrew.h
    ${GpuMesh_SRC_DIR}/DataStructures/OptionMap.h
    ${GpuMesh_SRC_DIR}/DataStructures/OptimizationPlot.h
    ${GpuMesh_SRC_DIR}/DataStructures/Tetrahedron.h
    ${GpuMesh_SRC_DIR}/DataStructures/TetList.h
    ${GpuMesh_SRC_DIR}/DataStructures/TetPool.h
    ${GpuMesh_SRC_DIR}/DataStructures/Triangle.h
    ${GpuMesh_SRC_DIR}/DataStructures/TriSet.h
    ${GpuMesh_SRC_DIR}/DataStructures/VertexAccum.h)

SET(GpuMesh_DISCRETIZERS_HEADERS
    ${GpuMesh_SRC_DIR}/Discretizers/AbstractDiscretizer.h
    ${GpuMesh_SRC_DIR}/Discretizers/AnalyticDiscretizer.h
    ${GpuMesh_SRC_DIR}/Discretizers/UniformDiscretizer.h
    ${GpuMesh_SRC_DIR}/Discretizers/KdTreeDiscretizer.h
    ${GpuMesh_SRC_DIR}/Discretizers/DummyDiscretizer.h)

SET(GpuMesh_EVALUATORS_HEADERS
    ${GpuMesh_SRC_DIR}/Evaluators/AbstractEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/InsphereEdgeEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/MeanRatioEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/MetricConformityEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/SolidAngleEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/VolumeEdgeEvaluator.h)

SET(GpuMesh_MEASURERS_HEADERS
    ${GpuMesh_SRC_DIR}/Measurers/AbstractMeasurer.h
    ${GpuMesh_SRC_DIR}/Measurers/MetricFreeMeasurer.h
    ${GpuMesh_SRC_DIR}/Measurers/MetricWiseMeasurer.h)

SET(GpuMesh_MESHERS_HEADERS
    ${GpuMesh_SRC_DIR}/Meshers/AbstractMesher.h
    ${GpuMesh_SRC_DIR}/Meshers/CpuDelaunayMesher.h
    ${GpuMesh_SRC_DIR}/Meshers/CpuParametricMesher.h
    ${GpuMesh_SRC_DIR}/Meshers/DebugMesher.h)

SET(GpuMesh_RENDERERS_HEADERS
    ${GpuMesh_SRC_DIR}/Renderers/AbstractRenderer.h
    ${GpuMesh_SRC_DIR}/Renderers/BlindRenderer.h
    ${GpuMesh_SRC_DIR}/Renderers/ScaffoldRenderer.h
    ${GpuMesh_SRC_DIR}/Renderers/SurfacicRenderer.h
    ${GpuMesh_SRC_DIR}/Renderers/QualityGradientPainter.h)

SET(GpuMesh_SERIALIZATION_HEADERS
    ${GpuMesh_SRC_DIR}/Serialization/AbstractSerializer.h
    ${GpuMesh_SRC_DIR}/Serialization/AbstractDeserializer.h
    ${GpuMesh_SRC_DIR}/Serialization/JsonMeshTags.h
    ${GpuMesh_SRC_DIR}/Serialization/JsonSerializer.h
    ${GpuMesh_SRC_DIR}/Serialization/JsonDeserializer.h
    ${GpuMesh_SRC_DIR}/Serialization/StlSerializer.h)

SET(GpuMesh_VERTEXWISE_HEADERS
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/AbstractVertexWiseSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/SpringLaplaceSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/QualityLaplaceSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/LocalOptimisationSmoother.h)

SET(GpuMesh_ELEMENTWISE_HEADERS
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/AbstractElementWiseSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/GetmeSmoother.h)

SET(GpuMesh_SMOOTHERS_HEADERS
    ${GpuMesh_VERTEXWISE_HEADERS}
    ${GpuMesh_ELEMENTWISE_HEADERS}
    ${GpuMesh_SRC_DIR}/Smoothers/AbstractSmoother.h)

SET(GpuMesh_DIALOGS_HEADERS
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/StlSerializerDialog.h)

SET(GpuMesh_UITABS_HEADERS
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/MeshTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/EvaluateTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/SmoothTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/RenderTab.h)

SET(GpuMesh_USERINTERFACE_HEADERS
    ${GpuMesh_DIALOGS_HEADERS}
    ${GpuMesh_UITABS_HEADERS}
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.h
    ${GpuMesh_SRC_DIR}/UserInterface/SmoothingReport.h)

SET(GpuMesh_HEADERS
    ${GpuMesh_DATASTRUCTURES_HEADERS}
    ${GpuMesh_DISCRETIZERS_HEADERS}
    ${GpuMesh_EVALUATORS_HEADERS}
    ${GpuMesh_MESHERS_HEADERS}
    ${GpuMesh_RENDERERS_HEADERS}
    ${GpuMesh_SERIALIZATION_HEADERS}
    ${GpuMesh_SMOOTHERS_HEADERS}
    ${GpuMesh_USERINTERFACE_HEADERS}
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.h)


## Sources ##

# All the source files #
SET(GpuMesh_DATASTRUCTURES_SOURCES
    ${GpuMesh_SRC_DIR}/DataStructures/Mesh.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/GpuMesh.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/MeshCrew.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/OptimizationPlot.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TetList.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TetPool.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TriSet.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/VertexAccum.cpp)

SET(GpuMesh_DISCRETIZERS_SOURCES
    ${GpuMesh_SRC_DIR}/Discretizers/AbstractDiscretizer.cpp
    ${GpuMesh_SRC_DIR}/Discretizers/AnalyticDiscretizer.cpp
    ${GpuMesh_SRC_DIR}/Discretizers/UniformDiscretizer.cpp
    ${GpuMesh_SRC_DIR}/Discretizers/KdTreeDiscretizer.cpp
    ${GpuMesh_SRC_DIR}/Discretizers/DummyDiscretizer.cpp)

SET(GpuMesh_EVALUATORS_SOURCES
    ${GpuMesh_SRC_DIR}/Evaluators/AbstractEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/InsphereEdgeEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/MeanRatioEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/MetricConformityEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/SolidAngleEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/VolumeEdgeEvaluator.cpp)

SET(GpuMesh_MEASURERS_SOURCES
    ${GpuMesh_SRC_DIR}/Measurers/AbstractMeasurer.cpp
    ${GpuMesh_SRC_DIR}/Measurers/MetricFreeMeasurer.cpp
    ${GpuMesh_SRC_DIR}/Measurers/MetricWiseMeasurer.cpp)

SET(GpuMesh_MESHERS_SOURCES
    ${GpuMesh_SRC_DIR}/Meshers/AbstractMesher.cpp
    ${GpuMesh_SRC_DIR}/Meshers/CpuDelaunayMesher.cpp
    ${GpuMesh_SRC_DIR}/Meshers/CpuParametricMesher.cpp
    ${GpuMesh_SRC_DIR}/Meshers/DebugMesher.cpp)

SET(GpuMesh_RENDERERS_SOURCES
    ${GpuMesh_SRC_DIR}/Renderers/AbstractRenderer.cpp
    ${GpuMesh_SRC_DIR}/Renderers/BlindRenderer.cpp
    ${GpuMesh_SRC_DIR}/Renderers/ScaffoldRenderer.cpp
    ${GpuMesh_SRC_DIR}/Renderers/SurfacicRenderer.cpp
    ${GpuMesh_SRC_DIR}/Renderers/QualityGradientPainter.cpp)

SET(GpuMesh_SERIALIZATION_SOURCES
    ${GpuMesh_SRC_DIR}/Serialization/AbstractSerializer.cpp
    ${GpuMesh_SRC_DIR}/Serialization/AbstractDeserializer.cpp
    ${GpuMesh_SRC_DIR}/Serialization/JsonMeshTags.cpp
    ${GpuMesh_SRC_DIR}/Serialization/JsonSerializer.cpp
    ${GpuMesh_SRC_DIR}/Serialization/JsonDeserializer.cpp
    ${GpuMesh_SRC_DIR}/Serialization/StlSerializer.cpp)


SET(GpuMesh_VERTEXWISE_SOURCES
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/AbstractVertexWiseSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/SpringLaplaceSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/QualityLaplaceSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/LocalOptimisationSmoother.cpp)

SET(GpuMesh_ELEMENTWISE_SOURCES
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/AbstractElementWiseSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/GetmeSmoother.cpp)

SET(GpuMesh_SMOOTHERS_SOURCES
    ${GpuMesh_VERTEXWISE_SOURCES}
    ${GpuMesh_ELEMENTWISE_SOURCES}
    ${GpuMesh_SRC_DIR}/Smoothers/AbstractSmoother.cpp)

SET(GpuMesh_UITABS_SOURCES
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/MeshTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/EvaluateTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/SmoothTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/RenderTab.cpp)

SET(GpuMesh_DIALOGS_HEADERS
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/StlSerializerDialog.cpp)

SET(GpuMesh_USERINTERFACE_SOURCES
    ${GpuMesh_DIALOGS_HEADERS}
    ${GpuMesh_UITABS_SOURCES}
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/SmoothingReport.cpp)

SET(GpuMesh_SOURCES
    ${GpuMesh_DATASTRUCTURES_SOURCES}
    ${GpuMesh_DISCRETIZERS_SOURCES}
    ${GpuMesh_EVALUATORS_SOURCES}
    ${GpuMesh_MEASURERS_SOURCES}
    ${GpuMesh_MESHERS_SOURCES}
    ${GpuMesh_RENDERERS_SOURCES}
    ${GpuMesh_SERIALIZATION_SOURCES}
    ${GpuMesh_SMOOTHERS_SOURCES}
    ${GpuMesh_USERINTERFACE_SOURCES}
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.cpp
    ${GpuMesh_SRC_DIR}/main.cpp)



## UI
SET(GpuMesh_UI_FILES
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.ui
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/StlSerializerDialog.ui)
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

SET(GpuMesh_DISCRETIZING_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Discretizing/Base.glsl
    ${GpuMesh_SHADER_DIR}/compute/Discretizing/Dummy.glsl
    ${GpuMesh_SHADER_DIR}/compute/Discretizing/Analytic.glsl
    ${GpuMesh_SHADER_DIR}/compute/Discretizing/KdTree.glsl
    ${GpuMesh_SHADER_DIR}/compute/Discretizing/Uniform.glsl)

SET(GpuMesh_EVALUATING_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Evaluating/Base.glsl
    ${GpuMesh_SHADER_DIR}/compute/Evaluating/Evaluate.glsl
    ${GpuMesh_SHADER_DIR}/compute/Evaluating/InsphereEdge.glsl
    ${GpuMesh_SHADER_DIR}/compute/Evaluating/MeanRatio.glsl
    ${GpuMesh_SHADER_DIR}/compute/Evaluating/MetricConformity.glsl
    ${GpuMesh_SHADER_DIR}/compute/Evaluating/SolidAngle.glsl
    ${GpuMesh_SHADER_DIR}/compute/Evaluating/VolumeEdge.glsl)

SET(GpuMesh_MEASURING_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Measuring/Base.glsl
    ${GpuMesh_SHADER_DIR}/compute/Measuring/MetricFree.glsl
    ${GpuMesh_SHADER_DIR}/compute/Measuring/MetricWise.glsl)

SET(GpuMesh_RENDERING_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Rendering/QualityGradient.glsl)

SET(GpuMesh_ELEMENTWISE_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/ElementWise/SmoothElements.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/ElementWise/UpdateVertices.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/ElementWise/VertexAccum.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/ElementWise/GETMe.glsl)

SET(GpuMesh_VERTEXWISE_SHADERS
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/VertexWise/SmoothVertices.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/VertexWise/SpringLaplace.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/VertexWise/QualityLaplace.glsl
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/VertexWise/LocalOptimisation.glsl)

SET(GpuMesh_SMOOTHING_SHADERS
    ${GpuMesh_ELEMENTWISE_SHADERS}
    ${GpuMesh_VERTEXWISE_SHADERS}
    ${GpuMesh_SHADER_DIR}/compute/Smoothing/Utils.glsl)

SET(GpuMesh_COMPUTE_SHADERS
    ${GpuMesh_BOUNDARY_SHADERS}
    ${GpuMesh_DISCRETIZING_SHADERS}
    ${GpuMesh_MEASURING_SHADERS}
    ${GpuMesh_EVALUATING_SHADERS}
    ${GpuMesh_RENDERING_SHADERS}
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
