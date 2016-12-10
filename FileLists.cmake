## Headers ##

# All the header files #
SET(GpuMesh_CONSTRAINTS_HEADERS
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/AbstractConstraint.h
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/VertexConstraint.h
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/EdgeConstraint.h
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/FaceConstraint.h
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/VolumeConstraint.h)

SET(GpuMesh_BOUNDARIES_HEADERS
    ${GpuMesh_CONSTRAINTS_HEADERS}
    ${GpuMesh_SRC_DIR}/Boundaries/AbstractBoundary.h
    ${GpuMesh_SRC_DIR}/Boundaries/BoundaryFree.h
    ${GpuMesh_SRC_DIR}/Boundaries/BoxBoundary.h
    ${GpuMesh_SRC_DIR}/Boundaries/PipeBoundary.h
    ${GpuMesh_SRC_DIR}/Boundaries/ShellBoundary.h
    ${GpuMesh_SRC_DIR}/Boundaries/SphereBoundary.h
    ${GpuMesh_SRC_DIR}/Boundaries/TetBoundary.h)

SET(GpuMesh_DATASTRUCTURES_HEADERS
    ${GpuMesh_SRC_DIR}/DataStructures/Mesh.h
    ${GpuMesh_SRC_DIR}/DataStructures/GpuMesh.h
    ${GpuMesh_SRC_DIR}/DataStructures/MeshCrew.h
    ${GpuMesh_SRC_DIR}/DataStructures/NodeGroups.h
    ${GpuMesh_SRC_DIR}/DataStructures/OptionMap.h
    ${GpuMesh_SRC_DIR}/DataStructures/OptimizationPlot.h
    ${GpuMesh_SRC_DIR}/DataStructures/Schedule.h
    ${GpuMesh_SRC_DIR}/DataStructures/Tetrahedralizer.h
    ${GpuMesh_SRC_DIR}/DataStructures/Tetrahedron.h
    ${GpuMesh_SRC_DIR}/DataStructures/TetList.h
    ${GpuMesh_SRC_DIR}/DataStructures/TetPool.h
    ${GpuMesh_SRC_DIR}/DataStructures/Triangle.h
    ${GpuMesh_SRC_DIR}/DataStructures/TriSet.h
    ${GpuMesh_SRC_DIR}/DataStructures/QualityHistogram.h)

SET(GpuMesh_SAMPLERS_HEADERS
    ${GpuMesh_SRC_DIR}/Samplers/AbstractSampler.h
    ${GpuMesh_SRC_DIR}/Samplers/AnalyticSampler.h
    ${GpuMesh_SRC_DIR}/Samplers/TextureSampler.h
    ${GpuMesh_SRC_DIR}/Samplers/KdTreeSampler.h
    ${GpuMesh_SRC_DIR}/Samplers/UniformSampler.h
    ${GpuMesh_SRC_DIR}/Samplers/LocalSampler.h)

SET(GpuMesh_EVALUATORS_HEADERS
    ${GpuMesh_SRC_DIR}/Evaluators/AbstractEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/MeanRatioEvaluator.h
    ${GpuMesh_SRC_DIR}/Evaluators/MetricConformityEvaluator.h)

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
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/GradientDescentSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/MultiPosGradDsntSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/PatchGradDsntSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/NelderMeadSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/SpawnSearchSmoother.h)

SET(GpuMesh_ELEMENTWISE_HEADERS
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/AbstractElementWiseSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/GetmeSmoother.h
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/VertexAccum.h)

SET(GpuMesh_SMOOTHERS_HEADERS
    ${GpuMesh_VERTEXWISE_HEADERS}
    ${GpuMesh_ELEMENTWISE_HEADERS}
    ${GpuMesh_SRC_DIR}/Smoothers/AbstractSmoother.h)

SET(GpuMesh_TOPOLOGISTS_HEADERS
    ${GpuMesh_SRC_DIR}/Topologists/AbstractTopologist.h
    ${GpuMesh_SRC_DIR}/Topologists/BatrTopologist.h)

SET(GpuMesh_DIALOGS_HEADERS
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/StlSerializerDialog.h
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/ConfigComparator.h)

SET(GpuMesh_UITABS_HEADERS
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/MeshTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/EvaluateTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/OptimizeTab.h
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/RenderTab.h)

SET(GpuMesh_USERINTERFACE_HEADERS
    ${GpuMesh_DIALOGS_HEADERS}
    ${GpuMesh_UITABS_HEADERS}
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.h
    ${GpuMesh_SRC_DIR}/UserInterface/SmoothingReport.h)

SET(GpuMesh_HEADERS
    ${GpuMesh_BOUNDARIES_HEADERS}
    ${GpuMesh_DATASTRUCTURES_HEADERS}
    ${GpuMesh_SAMPLERS_HEADERS}
    ${GpuMesh_EVALUATORS_HEADERS}
    ${GpuMesh_MESHERS_HEADERS}
    ${GpuMesh_RENDERERS_HEADERS}
    ${GpuMesh_SERIALIZATION_HEADERS}
    ${GpuMesh_SMOOTHERS_HEADERS}
    ${GpuMesh_TOPOLOGISTS_HEADERS}
    ${GpuMesh_USERINTERFACE_HEADERS}
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.h)


## Sources ##

# All the source files #
SET(GpuMesh_CONSTRAINTS_HEADERS
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/AbstractConstraint.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/VertexConstraint.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/EdgeConstraint.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/FaceConstraint.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/Constraints/VolumeConstraint.cpp)

SET(GpuMesh_BOUNDARIES_SOURCES
    ${GpuMesh_CONSTRAINTS_HEADERS}
    ${GpuMesh_SRC_DIR}/Boundaries/AbstractBoundary.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/BoundaryFree.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/BoxBoundary.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/PipeBoundary.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/ShellBoundary.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/SphereBoundary.cpp
    ${GpuMesh_SRC_DIR}/Boundaries/TetBoundary.cpp)

SET(GpuMesh_DATASTRUCTURES_SOURCES
    ${GpuMesh_SRC_DIR}/DataStructures/Mesh.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/GpuMesh.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/MeshCrew.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/NodeGroups.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/OptimizationPlot.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/Schedule.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TetList.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TetPool.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/TriSet.cpp
    ${GpuMesh_SRC_DIR}/DataStructures/QualityHistogram.cpp)

SET(GpuMesh_SAMPLERS_SOURCES
    ${GpuMesh_SRC_DIR}/Samplers/AbstractSampler.cpp
    ${GpuMesh_SRC_DIR}/Samplers/AnalyticSampler.cpp
    ${GpuMesh_SRC_DIR}/Samplers/TextureSampler.cpp
    ${GpuMesh_SRC_DIR}/Samplers/KdTreeSampler.cpp
    ${GpuMesh_SRC_DIR}/Samplers/UniformSampler.cpp
    ${GpuMesh_SRC_DIR}/Samplers/LocalSampler.cpp)

SET(GpuMesh_EVALUATORS_SOURCES
    ${GpuMesh_SRC_DIR}/Evaluators/AbstractEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/MeanRatioEvaluator.cpp
    ${GpuMesh_SRC_DIR}/Evaluators/MetricConformityEvaluator.cpp)

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
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/GradientDescentSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/MultiPosGradDsntSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/PatchGradDsntSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/NelderMeadSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/VertexWise/SpawnSearchSmoother.cpp)

SET(GpuMesh_ELEMENTWISE_SOURCES
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/AbstractElementWiseSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/GetmeSmoother.cpp
    ${GpuMesh_SRC_DIR}/Smoothers/ElementWise/VertexAccum.cpp)

SET(GpuMesh_SMOOTHERS_SOURCES
    ${GpuMesh_VERTEXWISE_SOURCES}
    ${GpuMesh_ELEMENTWISE_SOURCES}
    ${GpuMesh_SRC_DIR}/Smoothers/AbstractSmoother.cpp)

SET(GpuMesh_TOPOLOGISTS_SOURCES
    ${GpuMesh_SRC_DIR}/Topologists/AbstractTopologist.cpp
    ${GpuMesh_SRC_DIR}/Topologists/BatrTopologist.cpp)

SET(GpuMesh_UITABS_SOURCES
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/MeshTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/EvaluateTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/OptimizeTab.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Tabs/RenderTab.cpp)

SET(GpuMesh_DIALOGS_HEADERS
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/StlSerializerDialog.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/ConfigComparator.cpp)

SET(GpuMesh_USERINTERFACE_SOURCES
    ${GpuMesh_DIALOGS_HEADERS}
    ${GpuMesh_UITABS_SOURCES}
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.cpp
    ${GpuMesh_SRC_DIR}/UserInterface/SmoothingReport.cpp)

SET(GpuMesh_SOURCES
    ${GpuMesh_BOUNDARIES_SOURCES}
    ${GpuMesh_DATASTRUCTURES_SOURCES}
    ${GpuMesh_SAMPLERS_SOURCES}
    ${GpuMesh_EVALUATORS_SOURCES}
    ${GpuMesh_MEASURERS_SOURCES}
    ${GpuMesh_MESHERS_SOURCES}
    ${GpuMesh_RENDERERS_SOURCES}
    ${GpuMesh_SERIALIZATION_SOURCES}
    ${GpuMesh_SMOOTHERS_SOURCES}
    ${GpuMesh_TOPOLOGISTS_SOURCES}
    ${GpuMesh_USERINTERFACE_SOURCES}
    ${GpuMesh_SRC_DIR}/GpuMeshCharacter.cpp
    ${GpuMesh_SRC_DIR}/main.cpp)



## UI
SET(GpuMesh_UI_FILES
    ${GpuMesh_SRC_DIR}/UserInterface/MainWindow.ui
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/StlSerializerDialog.ui
    ${GpuMesh_SRC_DIR}/UserInterface/Dialogs/ConfigComparator.ui)
QT5_WRAP_UI(GpuMesh_UI_SRCS ${GpuMesh_UI_FILES})



## Resrouces ##

# GLSL directory
SET(GpuMesh_GLSL_DIR
    ${GpuMesh_SRC_DIR}/resources/glsl)

# Genereic shaders
SET(GpuMesh_GENERIC_SHADERS
    ${GpuMesh_GLSL_DIR}/generic/QualityLut.glsl
    ${GpuMesh_GLSL_DIR}/generic/Lighting.glsl)

# Vertex shaders
SET(GpuMesh_VERTEX_SHADERS
    ${GpuMesh_GLSL_DIR}/vertex/Shadow.vert
    ${GpuMesh_GLSL_DIR}/vertex/LitMesh.vert
    ${GpuMesh_GLSL_DIR}/vertex/UnlitMesh.vert
    ${GpuMesh_GLSL_DIR}/vertex/ScaffoldJoint.vert
    ${GpuMesh_GLSL_DIR}/vertex/ScaffoldTube.vert
    ${GpuMesh_GLSL_DIR}/vertex/Wireframe.vert
    ${GpuMesh_GLSL_DIR}/vertex/BoldEdge.vert
    ${GpuMesh_GLSL_DIR}/vertex/Bloom.vert
    ${GpuMesh_GLSL_DIR}/vertex/Filter.vert)

# Fragment shaders
SET(GpuMesh_FRAGMENT_SHADERS
    ${GpuMesh_GLSL_DIR}/fragment/Shadow.frag
    ${GpuMesh_GLSL_DIR}/fragment/LitMesh.frag
    ${GpuMesh_GLSL_DIR}/fragment/UnlitMesh.frag
    ${GpuMesh_GLSL_DIR}/fragment/ScaffoldJoint.frag
    ${GpuMesh_GLSL_DIR}/fragment/ScaffoldTube.frag
    ${GpuMesh_GLSL_DIR}/fragment/Wireframe.frag
    ${GpuMesh_GLSL_DIR}/fragment/BoldEdge.frag
    ${GpuMesh_GLSL_DIR}/fragment/BloomBlur.frag
    ${GpuMesh_GLSL_DIR}/fragment/BloomBlend.frag
    ${GpuMesh_GLSL_DIR}/fragment/Gradient.frag
    ${GpuMesh_GLSL_DIR}/fragment/Screen.frag
    ${GpuMesh_GLSL_DIR}/fragment/Brush.frag
    ${GpuMesh_GLSL_DIR}/fragment/Grain.frag)

# Compute shaders
SET(GpuMesh_DISCRETIZING_SHADERS
    ${GpuMesh_GLSL_DIR}/compute/Sampling/Base.glsl
    ${GpuMesh_GLSL_DIR}/compute/Sampling/Uniform.glsl
    ${GpuMesh_GLSL_DIR}/compute/Sampling/Analytic.glsl
    ${GpuMesh_GLSL_DIR}/compute/Sampling/KdTree.glsl
    ${GpuMesh_GLSL_DIR}/compute/Sampling/Local.glsl
    ${GpuMesh_GLSL_DIR}/compute/Sampling/Texture.glsl)

SET(GpuMesh_EVALUATING_SHADERS
    ${GpuMesh_GLSL_DIR}/compute/Evaluating/Base.glsl
    ${GpuMesh_GLSL_DIR}/compute/Evaluating/Evaluate.glsl
    ${GpuMesh_GLSL_DIR}/compute/Evaluating/MeanRatio.glsl
    ${GpuMesh_GLSL_DIR}/compute/Evaluating/MetricConformity.glsl)

SET(GpuMesh_MEASURING_SHADERS
    ${GpuMesh_GLSL_DIR}/compute/Measuring/Base.glsl
    ${GpuMesh_GLSL_DIR}/compute/Measuring/MetricFree.glsl
    ${GpuMesh_GLSL_DIR}/compute/Measuring/MetricWise.glsl)

SET(GpuMesh_RENDERING_SHADERS
    ${GpuMesh_GLSL_DIR}/compute/Rendering/QualityGradient.glsl)

SET(GpuMesh_ELEMENTWISE_SHADERS
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/ElementWise/SmoothElements.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/ElementWise/UpdateVertices.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/ElementWise/VertexAccum.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/ElementWise/GETMe.glsl)

SET(GpuMesh_VERTEXWISE_SHADERS
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/SmoothVertices.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/SpringLaplace.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/QualityLaplace.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/GradientDescent.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/MultiPosGradDsnt.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/PatchGradDsnt.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/NelderMead.glsl
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/VertexWise/SpawnSearch.glsl)

SET(GpuMesh_SMOOTHING_SHADERS
    ${GpuMesh_ELEMENTWISE_SHADERS}
    ${GpuMesh_VERTEXWISE_SHADERS}
    ${GpuMesh_GLSL_DIR}/compute/Smoothing/Utils.glsl)

SET(GpuMesh_COMPUTE_SHADERS
    ${GpuMesh_DISCRETIZING_SHADERS}
    ${GpuMesh_MEASURING_SHADERS}
    ${GpuMesh_EVALUATING_SHADERS}
    ${GpuMesh_RENDERING_SHADERS}
    ${GpuMesh_SMOOTHING_SHADERS}
    ${GpuMesh_GLSL_DIR}/compute/Mesh.glsl)

# GLSL sources
SET(GpuMesh_GLSL_SOURCES
    ${GpuMesh_GENERIC_SHADERS}
    ${GpuMesh_VERTEX_SHADERS}
    ${GpuMesh_FRAGMENT_SHADERS}
    ${GpuMesh_COMPUTE_SHADERS})


# CUDA directory
SET(GpuMesh_CUDA_DIR
    ${GpuMesh_SRC_DIR}/resources/cuda)

# Compute shaders
SET(GpuMesh_DISCRETIZING_CUDA
    ${GpuMesh_CUDA_DIR}/Sampling/Base.cuh
    ${GpuMesh_CUDA_DIR}/Sampling/Base.cu
    ${GpuMesh_CUDA_DIR}/Sampling/Uniform.cu
    ${GpuMesh_CUDA_DIR}/Sampling/Analytic.cu
    ${GpuMesh_CUDA_DIR}/Sampling/KdTree.cu
    ${GpuMesh_CUDA_DIR}/Sampling/Local.cu
    ${GpuMesh_CUDA_DIR}/Sampling/Texture.cu)

SET(GpuMesh_EVALUATING_CUDA
    ${GpuMesh_CUDA_DIR}/Evaluating/Base.cuh
    ${GpuMesh_CUDA_DIR}/Evaluating/Base.cu
    ${GpuMesh_CUDA_DIR}/Evaluating/Evaluate.cu
    ${GpuMesh_CUDA_DIR}/Evaluating/MeanRatio.cu
    ${GpuMesh_CUDA_DIR}/Evaluating/MetricConformity.cu)

SET(GpuMesh_MEASURING_CUDA
    ${GpuMesh_CUDA_DIR}/Measuring/Base.cuh
    ${GpuMesh_CUDA_DIR}/Measuring/Base.cu
    ${GpuMesh_CUDA_DIR}/Measuring/MetricFree.cu
    ${GpuMesh_CUDA_DIR}/Measuring/MetricWise.cu)

SET(GpuMesh_ELEMENTWISE_CUDA
    ${GpuMesh_CUDA_DIR}/Smoothing/ElementWise/Base.cuh
    ${GpuMesh_CUDA_DIR}/Smoothing/ElementWise/SmoothElements.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/ElementWise/UpdateVertices.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/ElementWise/VertexAccum.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/ElementWise/GETMe.cu)

SET(GpuMesh_VERTEXWISE_CUDA
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/Base.cuh
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/SmoothVertices.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/SpringLaplace.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/QualityLaplace.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/GradientDescent.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/MultiPosGradDsnt.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/PatchGradDsnt.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/NelderMead.cu
    ${GpuMesh_CUDA_DIR}/Smoothing/VertexWise/SpawnSearch.cu)

SET(GpuMesh_SMOOTHING_CUDA
    ${GpuMesh_ELEMENTWISE_CUDA}
    ${GpuMesh_VERTEXWISE_CUDA}
    ${GpuMesh_CUDA_DIR}/Smoothing/Utils.cu)

# CUDA sources
SET(GpuMesh_CUDA_SOURCES
    ${GpuMesh_DISCRETIZING_CUDA}
    ${GpuMesh_EVALUATING_CUDA}
    ${GpuMesh_MEASURING_CUDA}
    ${GpuMesh_SMOOTHING_CUDA}
    ${GpuMesh_CUDA_DIR}/Mesh.cuh
    ${GpuMesh_CUDA_DIR}/Mesh.cu)



# Textures
SET(GpuMesh_TEXTURE_DIR
    ${GpuMesh_SRC_DIR}/resources/textures)

# Background
SET(GpuMesh_BACKGROUND_TEX
    ${GpuMesh_TEXTURE_DIR}/Filter.png)


# Qrc File
QT5_ADD_RESOURCES(GpuMesh_RESOURCES
    ${GpuMesh_SRC_DIR}/resources/GpuMesh.qrc)


# Doc
SET(GpuMesh_DOC
    ${GpuMesh_SRC_DIR}/doc/Mémoire/fixnewline.py)


## Global ##
SET(GpuMesh_CONFIG_FILES
    ${GpuMesh_SRC_DIR}/CMakeLists.txt
    ${GpuMesh_SRC_DIR}/FileLists.cmake
    ${GpuMesh_SRC_DIR}/LibLists.cmake)
	
SET(GpuMesh_SRC_FILES
    ${GpuMesh_HEADERS}
    ${GpuMesh_SOURCES}
    ${GpuMesh_UI_SRCS}
    ${GpuMesh_GLSL_SOURCES}
    ${GpuMesh_CUDA_SOURCES}
    ${GpuMesh_BACKGROUND_TEX}
    ${GpuMesh_RESOURCES}
    ${GpuMesh_CONFIG_FILES}
    ${GpuMesh_MOC_CPP_FILES}
    ${GpuMesh_DOC})
