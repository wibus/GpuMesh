#include "GpuMeshCharacter.h"

#include <sstream>
#include <iostream>

#include <GLM/gtx/transform.hpp>
#include <GLM/gtc/matrix_transform.hpp>

#include <QFileInfo>

#include <CellarWorkbench/Misc/StringUtils.h>
#include <CellarWorkbench/Misc/Log.h>

#include <PropRoom2D/Team/AbstractTeam.h>

#include <Scaena/Play/Play.h>
#include <Scaena/Play/View.h>
#include <Scaena/StageManagement/Event/KeyboardEvent.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>
#include <Scaena/StageManagement/Event/StageTime.h>

#include "DataStructures/GpuMesh.h"
#include "Samplers/AnalyticSampler.h"
#include "Samplers/UniformSampler.h"
#include "Samplers/TextureSampler.h"
#include "Samplers/KdTreeSampler.h"
#include "Samplers/LocalSampler.h"
#include "Samplers/ComputedLocSampler.h"
#include "Samplers/ComputedTexSampler.h"
#include "Evaluators/MeanRatioEvaluator.h"
#include "Evaluators/MetricConformityEvaluator.h"
#include "Measurers/MetricFreeMeasurer.h"
#include "Measurers/MetricWiseMeasurer.h"
#include "Meshers/CpuDelaunayMesher.h"
#include "Meshers/CpuParametricMesher.h"
#include "Meshers/DebugMesher.h"
#include "Renderers/BlindRenderer.h"
#include "Renderers/ScaffoldRenderer.h"
#include "Renderers/SurfacicRenderer.h"
#include "Renderers/QualityGradientPainter.h"
#include "Serialization/CgnsDeserializer.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonDeserializer.h"
#include "Serialization/PieDeserializer.h"
#include "Serialization/StlSerializer.h"
#include "Smoothers/VertexWise/SpringLaplaceSmoother.h"
#include "Smoothers/VertexWise/QualityLaplaceSmoother.h"
#include "Smoothers/VertexWise/GradientDescentSmoother.h"
#include "Smoothers/VertexWise/MultiElemGradDsntSmoother.h"
#include "Smoothers/VertexWise/MultiPosGradDsntSmoother.h"
#include "Smoothers/VertexWise/PatchGradDsntSmoother.h"
#include "Smoothers/VertexWise/MultiElemNMSmoother.h"
#include "Smoothers/VertexWise/NelderMeadSmoother.h"
#include "Smoothers/VertexWise/SpawnSearchSmoother.h"
#include "Smoothers/ElementWise/GetmeSmoother.h"
#include "Topologists/AbstractTopologist.h"
#include "MastersTestSuite.h"

using namespace std;
using namespace cellar;
using namespace prop2;
using namespace scaena;


const std::string NO_METRIC_SAMPLING = "Uniform";
const glm::vec3 GpuMeshCharacter::nullVec = glm::vec3(0, 0, 0);
const glm::vec3 GpuMeshCharacter::upVec = glm::vec3(0, 0, 1);

GpuMeshCharacter::GpuMeshCharacter() :
    Character("GpuMeshChracter"),
    _camAzimuth(-glm::pi<float>() / 2.0),
    _camAltitude(0),
    _camDistance(4.0),
    _cameraMan(ECameraMan::Sphere),
    _lightAzimuth(-glm::pi<float>() * 3.5 / 8.0),
    _lightAltitude(-glm::pi<float>() * 2.0 / 4.0),
    _lightDistance(1.0),
    _cutAzimuth(0),
    _cutAltitude(-glm::pi<double>() / 2.0),
    _cutDistance(1.0e-8),
    _tetVisibility(true),
    _pyrVisibility(true),
    _priVisibility(true),
    _hexVisibility(true),
    _qualityCullingMin(-INFINITY),
    _qualityCullingMax(INFINITY),
    _displaySamplingMesh(false),
    _metricDiscretizationDepth(-1),
    _metricScaling(1.0),
    _metricAspectRatio(1.0),
    _mesh(new GpuMesh()),
    _meshCrew(new MeshCrew()),
    _mastersTestSuite(new MastersTestSuite(*this)),
    _computedMetricSmapler(new LocalSampler()),
    _availableMeshers("Available Meshers"),
    _availableSamplers("Available Samplers"),
    _availableEvaluators("Available Evaluators"),
    _availableSmoothers("Available Smoothers"),
    _availableRenderers("Available Renderers"),
    _availableCameraMen("Available Camera Men"),
    _availableCutTypes("Available Cut Types"),
    _availableSerializers("Available Mesh Serializers"),
    _availableDeserializers("Available Mesh Deserializers")
{
    _availableMeshers.setDefault("Delaunay");
    _availableMeshers.setContent({
        {string("Delaunay"),   shared_ptr<AbstractMesher>(new CpuDelaunayMesher())},
        {string("Parametric"), shared_ptr<AbstractMesher>(new CpuParametricMesher())},
        {string("Debug"),      shared_ptr<AbstractMesher>(new DebugMesher())},
    });

    _availableSamplers.setDefault("Analytic");
    _availableSamplers.setContent({
        {NO_METRIC_SAMPLING,     shared_ptr<AbstractSampler>(new UniformSampler())},
        {string("Analytic"),     shared_ptr<AbstractSampler>(new AnalyticSampler())},
        {string("Texture"),      shared_ptr<AbstractSampler>(new TextureSampler())},
        {string("Kd-Tree"),      shared_ptr<AbstractSampler>(new KdTreeSampler())},
        {string("Local"),        shared_ptr<AbstractSampler>(new LocalSampler())},
        {string("Computed Loc"), shared_ptr<AbstractSampler>(new ComputedLocSampler())},
        {string("Computed Tex"), shared_ptr<AbstractSampler>(new ComputedTexSampler())},
    });

    _availableEvaluators.setDefault("Metric Conformity");
    _availableEvaluators.setContent({
        {string("Mean Ratio"),        shared_ptr<AbstractEvaluator>(new MeanRatioEvaluator())},
        {string("Metric Conformity"), shared_ptr<AbstractEvaluator>(new MetricConformityEvaluator())},
    });

    _availableSmoothers.setDefault("Nelder-Mead");
    _availableSmoothers.setContent({
        {string("Spring Laplace"),   shared_ptr<AbstractSmoother>(new SpringLaplaceSmoother())},
        {string("Quality Laplace"),  shared_ptr<AbstractSmoother>(new QualityLaplaceSmoother())},
        {string("Gradient Descent"), shared_ptr<AbstractSmoother>(new GradientDescentSmoother())},
        {string("Multi Elem GD"),    shared_ptr<AbstractSmoother>(new MultiElemGradDsntSmoother())},
        {string("Multi Pos GD"),     shared_ptr<AbstractSmoother>(new MultiPosGradDsntSmoother())},
        {string("Patch GD"),         shared_ptr<AbstractSmoother>(new PatchGradDsntSmoother())},
        {string("Multi Elem NM"),    shared_ptr<AbstractSmoother>(new MultiElemNMSmoother())},
        {string("Nelder-Mead"),      shared_ptr<AbstractSmoother>(new NelderMeadSmoother())},
        {string("Spawn Search"),     shared_ptr<AbstractSmoother>(new SpawnSearchSmoother())},
        {string("GETMe"),            shared_ptr<AbstractSmoother>(new GetmeSmoother())},
    });

    _availableRenderers.setDefault("Surfacic");
    _availableRenderers.setContent({
        {string("Blind"),    shared_ptr<AbstractRenderer>(new BlindRenderer())},
        {string("Scaffold"), shared_ptr<AbstractRenderer>(new ScaffoldRenderer())},
        {string("Surfacic"), shared_ptr<AbstractRenderer>(new SurfacicRenderer())},
    });

    _availableCameraMen.setDefault("Sphere");
    _availableCameraMen.setContent({
        {string("Oblique"), ECameraMan::Oblique},
        {string("Sphere"),  ECameraMan::Sphere},
        {string("Free"),    ECameraMan::Free},
    });

    _availableCutTypes.setDefault("None");
    _availableCutTypes.setContent({
        {string("None"),              ECutType::None},
        {string("Virtual Plane"),     ECutType::VirtualPlane},
        {string("Physical Plane"),    ECutType::PhysicalPlane},
        {string("Inverted Elements"), ECutType::InvertedElements},
    });

    _availableSerializers.setDefault("json");
    _availableSerializers.setContent({
        {string("json"), shared_ptr<AbstractSerializer>(new JsonSerializer())},
        {string("stl"),  shared_ptr<AbstractSerializer>(new StlSerializer())},
    });

    _availableDeserializers.setDefault("json");
    _availableDeserializers.setContent({
        {string("cgns"), shared_ptr<AbstractDeserializer>(new CgnsDeserializer())},
        {string("json"), shared_ptr<AbstractDeserializer>(new JsonDeserializer())},
#ifdef ENABLE_PIRATE
        {string("pie"), shared_ptr<AbstractDeserializer>(new PieDeserializer())},
#endif
    });
}

void GpuMeshCharacter::enterStage()
{
    _fps = play().propTeam2D()->createTextHud();
    _fps->setHandlePosition(glm::dvec2(5, 5));
    _fps->setHorizontalAnchor(EHorizontalAnchor::LEFT);
    _fps->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _fps->setHeight(16);
    _fps->setIsVisible(false);

    _ups = play().propTeam2D()->createTextHud();
    _ups->setHandlePosition(glm::dvec2(5, 25));
    _ups->setHorizontalAnchor(EHorizontalAnchor::LEFT);
    _ups->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _ups->setHeight(16);
    _ups->setIsVisible(false);

    _qualityTitle = play().propTeam2D()->createTextHud();
    _qualityTitle->setHandlePosition(glm::dvec2(-120, 294));
    _qualityTitle->setHorizontalAnchor(EHorizontalAnchor::RIGHT);
    _qualityTitle->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _qualityTitle->setHeight(26);
    _qualityTitle->setText("Quality");

    _qualityMax = play().propTeam2D()->createTextHud();
    _qualityMax->setHandlePosition(glm::dvec2(-130, 268));
    _qualityMax->setHorizontalAnchor(EHorizontalAnchor::RIGHT);
    _qualityMax->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _qualityMax->setHeight(16);
    _qualityMax->setText("Max -");

    _qualityMin = play().propTeam2D()->createTextHud();
    _qualityMin->setHandlePosition(glm::dvec2(-130, 51));
    _qualityMin->setHorizontalAnchor(EHorizontalAnchor::RIGHT);
    _qualityMin->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _qualityMin->setHeight(16);
    _qualityMin->setText("Min -");

    _qualityNeg = play().propTeam2D()->createTextHud();
    _qualityNeg->setHandlePosition(glm::dvec2(-130, 30));
    _qualityNeg->setHorizontalAnchor(EHorizontalAnchor::RIGHT);
    _qualityNeg->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _qualityNeg->setHeight(16);
    _qualityNeg->setText("Neg {");

    int LUT_WIDTH = 40;
    int LUT_HEIGHT = 256;
    _qualityLut = play().propTeam2D()->createImageHud();
    _qualityLut->setHandlePosition(glm::dvec2(-90, 20));
    _qualityLut->setHorizontalAnchor(EHorizontalAnchor::RIGHT);
    _qualityLut->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _qualityLut->setSize(glm::dvec2(LUT_WIDTH, LUT_HEIGHT));

    QualityGradientPainter painter;
    _qualityLut->setImageName(
        painter.generate(
            LUT_WIDTH,
            LUT_HEIGHT,
            0.15 * LUT_HEIGHT));



    // Assess evaluators validy
    UniformSampler verifSampler;
    MetricWiseMeasurer verifMeasurer;
    for(auto evalName : _availableEvaluators.details().options)
    {
        std::shared_ptr<AbstractEvaluator> evaluator;
        if(_availableEvaluators.select(evalName, evaluator))
        {
            evaluator->assessMeasureValidy(verifSampler, verifMeasurer);
        }
    }

    _cameraManFree.reset(new CameraManFree(play().view()->camera3D(), false));
    _cameraManFree->setPosition(glm::vec3(0, 0, _camDistance));
    _cameraManFree->setOrientation( glm::pi<float>() * 0.5f,
                                   -glm::pi<float>() * 0.48f);
    play().view()->camera3D()->registerObserver(*this);
    setupInstalledRenderer();
    _meshCrew->initialize(*_mesh);


    GLint variableGroupSize[3] = {0, 0, 0};
    glGetIntegeri_v(GL_MAX_COMPUTE_VARIABLE_GROUP_SIZE_ARB,
                  0, &variableGroupSize[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_VARIABLE_GROUP_SIZE_ARB,
                  1, &variableGroupSize[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_VARIABLE_GROUP_SIZE_ARB,
                  2, &variableGroupSize[2]);

    getLog().postMessage(new Message('I', false,
        "Max variable group size: (" +
                to_string(variableGroupSize[0]) + ", " +
                to_string(variableGroupSize[1]) + ", " +
                to_string(variableGroupSize[2]) + ")"
                , "GpuMeshCharacter"));

    GLint variableInvocCount = 0;
    glGetIntegerv(GL_MAX_COMPUTE_VARIABLE_GROUP_INVOCATIONS_ARB,
                  &variableInvocCount);

    getLog().postMessage(new Message('I', false,
        "Max variable invocation count: " +
                to_string(variableInvocCount),
        "GpuMeshCharacter"));
}

void GpuMeshCharacter::beginStep(const StageTime &time)
{
    _ups->setText("UPS: " + to_string(time.framesPerSecond()));

    std::shared_ptr<SynchronousKeyboard> keyboard = play().synchronousKeyboard();
    std::shared_ptr<SynchronousMouse> mouse = play().synchronousMouse();

    if(keyboard->isNonAsciiPressed(ENonAscii::SHIFT))
    {
        // Cut plane management
        if(mouse->buttonIsPressed(EMouseButton::LEFT))
        {
            moveCutPlane(_cutAzimuth - mouse->displacement().x / 200.0f,
                         _cutAltitude - mouse->displacement().y / 200.0f,
                         _cutDistance);
        }
        else if(mouse->degreeDelta() != 0)
        {
            moveCutPlane(_cutAzimuth,
                         _cutAltitude,
                         _cutDistance + mouse->degreeDelta() / 800.0f);
        }
    }
    else if(keyboard->isNonAsciiPressed(ENonAscii::CTRL))
    {
        // Cut plane management
        if(mouse->buttonIsPressed(EMouseButton::LEFT))
        {
            moveLight(_lightAzimuth - mouse->displacement().x / 200.0f,
                      _lightAltitude - mouse->displacement().y / 200.0f,
                      _lightDistance);
        }
        else if(mouse->degreeDelta() != 0)
        {
            moveLight(_lightAzimuth,
                      _lightAltitude,
                      _lightDistance + mouse->degreeDelta() / 800.0f);
        }
    }
    else
    {
        // Camera management
        if(_cameraMan == ECameraMan::Oblique)
            ;
        else if(_cameraMan == ECameraMan::Sphere)
        {
            if(mouse->buttonIsPressed(EMouseButton::LEFT))
            {
                moveSphereCamera(_camAzimuth - mouse->displacement().x / 100.0f,
                           _camAltitude - mouse->displacement().y / 100.0f,
                           _camDistance);
            }
            else if(mouse->degreeDelta() != 0)
            {
                moveSphereCamera(_camAzimuth,
                           _camAltitude,
                           _camDistance + mouse->degreeDelta() / 80.0f);
            }
        }
        else if(_cameraMan == ECameraMan::Free)
        {
            float velocity  = 0.5f * time.elapsedTime();
            float turnSpeed = 0.004f;
            bool updated = false;

            if(keyboard->isAsciiPressed('w'))
            {
                _cameraManFree->forward(velocity);
                updated = true;
            }
            if(keyboard->isAsciiPressed('s'))
            {
                _cameraManFree->forward(-velocity);
                updated = true;
            }
            if(keyboard->isAsciiPressed('a'))
            {
                _cameraManFree->sideward(-velocity / 3.0f);
                updated = true;
            }
            if(keyboard->isAsciiPressed('d'))
            {
                _cameraManFree->sideward(velocity / 3.0f);
                updated = true;
            }

            if(mouse->displacement() != glm::ivec2(0, 0) &&
               mouse->buttonIsPressed(EMouseButton::LEFT))
            {
                _cameraManFree->pan( mouse->displacement().x * -turnSpeed);
                _cameraManFree->tilt(mouse->displacement().y * -turnSpeed);
                updated = true;
            }

            if(updated)
                _renderer->updateCamera(_cameraManFree->position());
        }
    }

    _renderer->handleInputs(*keyboard, *mouse);
}

void GpuMeshCharacter::draw(const shared_ptr<View>&, const StageTime& time)
{
    _fps->setText("UPS: " + to_string(time.framesPerSecond()));

    if(_displaySamplingMesh)
        _renderer->display(_meshCrew->sampler().debugMesh());
    else
        _renderer->display(*_mesh);
}

void GpuMeshCharacter::exitStage()
{
    tearDownInstalledRenderer();
    _meshCrew->terminate();
}

bool GpuMeshCharacter::keyPressEvent(const scaena::KeyboardEvent &event)
{
    return _renderer->handleKeyPress(event);
}

OptionMapDetails GpuMeshCharacter::availableMeshers() const
{
    return _availableMeshers.details();
}

OptionMapDetails GpuMeshCharacter::availableMeshModels(const string& mesherName) const
{
    std::shared_ptr<AbstractMesher> mesher;
    if(_availableMeshers.select(mesherName, mesher))
        return mesher->availableMeshModels();
    else
        return OptionMapDetails();
}

OptionMapDetails GpuMeshCharacter::availableSamplers() const
{
    return _availableSamplers.details();
}

OptionMapDetails GpuMeshCharacter::availableEvaluators() const
{
    return _availableEvaluators.details();
}

OptionMapDetails GpuMeshCharacter::availableEvaluatorImplementations(const string& evaluatorName) const
{
    std::shared_ptr<AbstractEvaluator> evaluator;
    if(_availableEvaluators.select(evaluatorName, evaluator))
        return evaluator->availableImplementations();
    else
        return OptionMapDetails();
}

OptionMapDetails GpuMeshCharacter::availableSmoothers() const
{
    return _availableSmoothers.details();
}

OptionMapDetails GpuMeshCharacter::availableSmootherImplementations(const string& smootherName) const
{
    std::shared_ptr<AbstractSmoother> smoother;
    if(_availableSmoothers.select(smootherName, smoother))
        return smoother->availableImplementations();
    else
        return OptionMapDetails();
}

OptionMapDetails GpuMeshCharacter::availableRenderers() const
{
    return _availableRenderers.details();
}

OptionMapDetails GpuMeshCharacter::availableShadings() const
{
    return _renderer->availableShadings();
}

OptionMapDetails GpuMeshCharacter::availableMastersTests() const
{
    return _mastersTestSuite->availableTests();
}

OptionMapDetails GpuMeshCharacter::availableCameraMen() const
{
    return _availableCameraMen.details();
}

OptionMapDetails GpuMeshCharacter::availableCutTypes() const
{
    return _availableCutTypes.details();
}

void GpuMeshCharacter::generateMesh(
        const std::string& mesherName,
        const std::string& modelName,
        size_t vertexCount)
{
    clearMesh();

    printStep("Mesh Generation "\
              ": mesher=" + mesherName +
              ", model=" + modelName +
              ", vertex count=" + to_string(vertexCount));

    std::shared_ptr<AbstractMesher> mesher;
    if(_availableMeshers.select(mesherName, mesher))
    {
        mesher->generateMesh(*_mesh, modelName, vertexCount);

        _mesh->modelName = modelName;
        _mesh->compileTopology();

        _computedMetricSmapler->buildBackgroundMesh(
            *_mesh, vector<MeshMetric>(getNodeCount(), MeshMetric()));

        updateSampling();
        updateMeshMeasures();
    }
}

size_t GpuMeshCharacter::getNodeCount() const
{
    return _mesh->verts.size();
}

void GpuMeshCharacter::clearMesh()
{
    _mesh->clear();
    _mesh->modelName = "";

    _computedMetricSmapler->buildBackgroundMesh(
        *_mesh, std::vector<MeshMetric>());

    updateSampling();
    updateMeshMeasures();
}

string fileExt(const std::string& fileName)
{
    QString qFileName(fileName.c_str());
    QString qExt = QFileInfo(qFileName).suffix();
    return qExt.toStdString();
}

void GpuMeshCharacter::saveMesh(
        const std::string& fileName)
{
    printStep("Saving mesh at " + fileName);

    string ext = fileExt(fileName);
    shared_ptr<AbstractSerializer> serializer;
    if(_availableSerializers.select(ext, serializer))
    {
        if(!serializer->serialize(fileName, *_meshCrew, *_mesh))
        {
            getLog().postMessage(new Message('E', false,
                "An error occured while saving the mesh.", "GpuMeshCharacter"));
        }
    }
}

bool GpuMeshCharacter::loadMesh(
        const std::string& fileName)
{
    printStep("Loading mesh from " + fileName);

    string ext = fileExt(fileName);
    shared_ptr<AbstractDeserializer> deserializer;
    if(_availableDeserializers.select(ext, deserializer))
    {
        clearMesh();

        std::vector<MeshMetric> metrics;
        if(deserializer->deserialize(fileName, *_mesh, metrics))
        {
            if(metrics.empty()) metrics.resize(getNodeCount(), MeshMetric());
            _computedMetricSmapler->buildBackgroundMesh(*_mesh, metrics);

            _mesh->compileTopology();

            updateSampling();
            updateMeshMeasures();

            return true;
        }
        else
        {
            getLog().postMessage(new Message('E', false,
                "An error occured while loading the mesh.", "GpuMeshCharacter"));
        }
    }

    return false;
}

void GpuMeshCharacter::evaluateMesh(
            const std::string& evaluatorName,
            const std::string& implementationName)
{
    printStep("Shape measure "\
              ": evalutor=" + evaluatorName +
              ", implementation=" + implementationName);

    std::shared_ptr<AbstractEvaluator> evaluator;
    if(_availableEvaluators.select(evaluatorName, evaluator))
    {
        QualityHistogram histogram;

        evaluator->setGlslThreadCount(_glslEvaluatorThreadCount);
        evaluator->setCudaThreadCount(_cudaEvaluatorThreadCount);

        evaluator->evaluateMesh(
            *_mesh,
            _meshCrew->sampler(),
            _meshCrew->measurer(),
            histogram,
            implementationName);

        getLog().postMessage(new Message('I', false,
            "Results "\
            ": min=" + to_string(histogram.minimumQuality()) +
            ", mean=" + to_string(histogram.harmonicMean()),
             "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::benchmarkEvaluator(
        map<string, double>& averageTimes,
        const std::string& evaluatorName,
        const map<string, int>& cycleCounts)
{
    printStep("Shape measure evaluation benchmark "\
              ": quality measure=" + evaluatorName);

    std::shared_ptr<AbstractEvaluator> evaluator;
    if(_availableEvaluators.select(evaluatorName, evaluator))
    {
        evaluator->setGlslThreadCount(_glslEvaluatorThreadCount);
        evaluator->setCudaThreadCount(_cudaEvaluatorThreadCount);

        evaluator->benchmark(
            *_mesh,
            _meshCrew->sampler(),
            _meshCrew->measurer(),
            cycleCounts,
            averageTimes);
    }
}

void GpuMeshCharacter::setMetricScaling(double scaling)
{
    getLog().postMessage(new Message('I', false,
        "Setting metric scaling factor (K) to " + to_string(scaling),
        "GpuMeshCharacter"));

    _metricScaling = scaling;

    updateSampling();
    updateMeshMeasures();
}

void GpuMeshCharacter::setMetricAspectRatio(double ratio)
{
    getLog().postMessage(new Message('I', false,
        "Setting metric aspection ratio factor (A) to " + to_string(ratio),
        "GpuMeshCharacter"));

    _metricAspectRatio = ratio;

    updateSampling();
    updateMeshMeasures();
}

void GpuMeshCharacter::setMetricDiscretizationDepth(int depth)
{
    getLog().postMessage(new Message('I', false,
        "Setting metric discretization depth " + to_string(depth),
        "GpuMeshCharacter"));

    _metricDiscretizationDepth = depth;

    updateSampling();
    updateMeshMeasures();
}

void GpuMeshCharacter::setGlslEvaluatorThreadCount(uint threadCount)
{
    getLog().postMessage(new Message('I', false,
        "Setting GLSL evaluator thread count to " + to_string(threadCount),
        "GpuMeshCharacter"));

    _glslEvaluatorThreadCount = threadCount;
}

void GpuMeshCharacter::setCudaEvaluatorThreadCount(uint threadCount)
{
    getLog().postMessage(new Message('I', false,
        "Setting CUDA evaluator thread count to " + to_string(threadCount),
        "GpuMeshCharacter"));

    _cudaEvaluatorThreadCount = threadCount;
}

void GpuMeshCharacter::setGlslSmootherThreadCount(uint threadCount)
{
    getLog().postMessage(new Message('I', false,
        "Setting GLSL smoother thread count to " + to_string(threadCount),
        "GpuMeshCharacter"));

    _glslSmootherThreadCount = threadCount;
}

void GpuMeshCharacter::setCudaSmootherThreadCount(uint threadCount)
{
    getLog().postMessage(new Message('I', false,
        "Setting CUDA smoother thread count to " + to_string(threadCount),
        "GpuMeshCharacter"));

    _cudaSmootherThreadCount = threadCount;
}

void GpuMeshCharacter::smoothMesh(
        const std::string& smootherName,
        const std::string& implementationName,
        const Schedule& schedule)
{
    printStep("Mesh Smoothing "\
              ": smoother=" + smootherName +
              ", implementation=" + implementationName);

    std::shared_ptr<AbstractSmoother> smoother;
    if(_availableSmoothers.select(smootherName, smoother))
    {
        OptimizationImpl impl;

        _meshCrew->evaluator().setGlslThreadCount(_glslEvaluatorThreadCount);
        _meshCrew->evaluator().setCudaThreadCount(_cudaEvaluatorThreadCount);
        smoother->setGlslThreadCount(_glslSmootherThreadCount);
        smoother->setCudaThreadCount(_cudaSmootherThreadCount);

        smoother->smoothMesh(
            *_mesh,
            *_meshCrew,
            implementationName,
            schedule,
            impl);

        updateSampling();
        updateMeshMeasures();
    }
}

void GpuMeshCharacter::benchmarkSmoothers(
        OptimizationPlot& plot,
        const Schedule& schedule,
        const std::vector<Configuration>& configurations)
{
    printStep("Smoothing benchmark "\
              ": " + _mesh->modelName);


    std::shared_ptr<Mesh> currMesh(new Mesh(*_mesh));
    std::shared_ptr<AbstractSampler> currSampler = _meshCrew->samplerPtr();

    std::shared_ptr<AbstractSampler> referenceSampler;
    if(currSampler->useComputedMetric())
    {
        referenceSampler = _computedMetricSmapler;
        _meshCrew->setSampler(*_mesh, referenceSampler);
    }
    else
    {
        _availableSamplers.select("Analytic", referenceSampler);
        _meshCrew->setSampler(*_mesh, referenceSampler);
        updateSampling();
    }

    QualityHistogram initialHistogram;
    _meshCrew->evaluator().evaluateMeshQualityThread(
        *_mesh, _meshCrew->sampler(), _meshCrew->measurer(),
        initialHistogram);

    _mesh->printPropperties(plot);
    plot.setMeshModelName(_mesh->modelName);
    plot.setInitialHistogram(initialHistogram);
    plot.setNodeGroups(_mesh->nodeGroups());


    _meshCrew->evaluator().setGlslThreadCount(_glslEvaluatorThreadCount);
    _meshCrew->evaluator().setCudaThreadCount(_cudaEvaluatorThreadCount);


    for(const Configuration& config : configurations)
    {
        std::shared_ptr<AbstractSampler> sampler;
        if(_availableSamplers.select(config.samplerName, sampler))
        {
            *_mesh = *currMesh;
            _meshCrew->setSampler(*_mesh, sampler);

            updateSampling();

            std::shared_ptr<AbstractSmoother> smoother;
            if(_availableSmoothers.select(config.smootherName, smoother))
            {
                OptimizationImpl impl;
                impl.configuration = config;
                impl.name = config.smootherName +
                    "(samp=" + config.samplerName +
                    ", impl=" + config.implementationName + ")";
                impl.isTopologicalOperationOn = schedule.topoOperationEnabled;

                smoother->setGlslThreadCount(_glslSmootherThreadCount);
                smoother->setCudaThreadCount(_cudaSmootherThreadCount);

                printStep("Benchmarking smoothing configuration: "\
                          ": " + impl.name);

                smoother->smoothMesh(
                    *_mesh,
                    *_meshCrew,
                    config.implementationName,
                    schedule,
                    impl);


                _meshCrew->setSampler(*_mesh, referenceSampler);
                _meshCrew->evaluator().evaluateMeshQualityThread(
                    *_mesh, _meshCrew->sampler(), _meshCrew->measurer(),
                    impl.finalHistogram);

                plot.addImplementation(impl);
            }
        }
    }

    _meshCrew->setSampler(*_mesh, currSampler);

    updateSampling();
    updateMeshMeasures();
}

void GpuMeshCharacter::restructureMesh(const Schedule& schedule)
{
    _meshCrew->topologist().restructureMesh(
        *_mesh, *_meshCrew, schedule);

    updateSampling();
    updateMeshMeasures();
}

void GpuMeshCharacter::disableAnisotropy()
{
    displaySamplingMesh(false);
    useSampler(NO_METRIC_SAMPLING);
}

void GpuMeshCharacter::displaySamplingMesh(bool display)
{
    _displaySamplingMesh = display;
    if(_renderer.get() != nullptr)
        _renderer->notifyMeshUpdate();

    if(!_displaySamplingMesh)
        _meshCrew->sampler().releaseDebugMesh();
}

void GpuMeshCharacter::useSampler(const std::string& samplerName)
{
    std::shared_ptr<AbstractSampler> sampler;
    if(_availableSamplers.select(samplerName, sampler))
    {
        _meshCrew->setSampler(*_mesh, sampler);

        updateSampling();
        updateMeshMeasures();
    }
}

void GpuMeshCharacter::useEvaluator(const std::string& evaluatorName)
{
    std::shared_ptr<AbstractEvaluator> evaluator;
    if(_availableEvaluators.select(evaluatorName, evaluator))
    {
        _meshCrew->setEvaluator(*_mesh, evaluator);
        updateMeshMeasures();
    }
}

void GpuMeshCharacter::useRenderer(const std::string& rendererName)
{
    std::shared_ptr<AbstractRenderer> renderer;
    if(_availableRenderers.select(rendererName, renderer))
        installRenderer(renderer);
}

void GpuMeshCharacter::useShading(const std::string& shadingName)
{
    _renderer->useShading(shadingName);
}

void GpuMeshCharacter::useCameraMan(const string& cameraManName)
{
    _availableCameraMen.select(cameraManName, _cameraMan);

    if(_meshCrew->initialized())
    {
        refreshCamera();
    }
}

void GpuMeshCharacter::useCutType(const std::string& cutTypeName)
{
    _availableCutTypes.select(cutTypeName, _cutType);

    if(_meshCrew->initialized())
    {
        _renderer->useCutType(_cutType);
    }
}

void GpuMeshCharacter::displayBackdrop(bool display)
{
    _displayBackdrop = display;

    if(_meshCrew->initialized())
    {
        _renderer->displayBackdrop(_displayBackdrop);
    }
}

void GpuMeshCharacter::setElementVisibility(bool tet, bool pyr, bool pri, bool hex)
{
    _tetVisibility = tet;
    _pyrVisibility = pyr;
    _priVisibility = pri;
    _hexVisibility = hex;

    if(_meshCrew->initialized())
    {
        _renderer->setElementVisibility(tet, pyr, pri, hex);
    }
}

void GpuMeshCharacter::setQualityCullingBounds(double min, double max)
{
    _qualityCullingMin = min;
    _qualityCullingMax = max;

    if(_meshCrew->initialized())
    {
        _renderer->setQualityCullingBounds(min, max);
    }
}

void GpuMeshCharacter::notify(cellar::CameraMsg& msg)
{
    if(msg.change == CameraMsg::EChange::VIEWPORT)
    {
        const glm::ivec2& viewport = msg.camera.viewport();

        glm::mat4 proj(1.0);

        if(_cameraMan == ECameraMan::Oblique)
        {
            glm::mat4 ortho = glm::ortho(
                -viewport.x / 200.0f, viewport.x / 200.0f,
                -viewport.y / 200.0f, viewport.y / 200.0f,
                -5.0f, 5.0f);

            glm::mat4 oblique(1.0);
            oblique[2][0] = -1.0 / tan(glm::pi<double>() * (110.0 / 180.0));
            oblique[2][1] = 1.0 / tan(glm::pi<double>() * (110.0 / 180.0));

            proj = ortho * oblique;

        }
        else
        {
            proj = glm::perspectiveFov(
                glm::pi<float>() / 6.0f,
                (float) viewport.x,
                (float) viewport.y,
                0.1f,
                12.0f);
        }

        msg.camera.updateProjection(proj);
    }
}

void GpuMeshCharacter::runMastersTests(
        QTextDocument& reportDocument,
        const vector<string>& tests)
{
    _mastersTestSuite->runTests(reportDocument, tests);
}

void GpuMeshCharacter::printStep(const std::string& stepDescription)
{
    getLog().postMessage(new Message('I', false, stepDescription, "GpuMeshCharacter"));
}

void GpuMeshCharacter::refreshCamera()
{
    play().view()->camera3D()->refresh();

    if(_cameraMan == ECameraMan::Oblique)
        moveObliqueCamera();
    else if(_cameraMan == ECameraMan::Sphere)
        moveSphereCamera(_camAzimuth, _camAltitude, _camDistance);
    else if(_cameraMan == ECameraMan::Free)
        moveFreeCamera();
}

void GpuMeshCharacter::updateMeshMeasures()
{
    if(_meshCrew->initialized())
    {
        size_t tetCount = _mesh->tets.size();
        for(size_t e=0; e < tetCount; ++e)
        {
            MeshTet& elem = _mesh->tets[e];
            elem.value = _meshCrew->evaluator().tetQuality(
                *_mesh,
                _meshCrew->sampler(),
                _meshCrew->measurer(),
                elem);
        }

        size_t priCount = _mesh->pris.size();
        for(size_t e=0; e < priCount; ++e)
        {
            MeshPri& elem = _mesh->pris[e];
            elem.value = _meshCrew->evaluator().priQuality(
                *_mesh,
                _meshCrew->sampler(),
                _meshCrew->measurer(),
                elem);
        }

        size_t hexCount = _mesh->hexs.size();
        for(size_t e=0; e < hexCount; ++e)
        {
            MeshHex& elem = _mesh->hexs[e];
            elem.value = _meshCrew->evaluator().hexQuality(
                *_mesh,
                _meshCrew->sampler(),
                _meshCrew->measurer(),
                elem);
        }

        if(_renderer.get() != nullptr)
            _renderer->notifyMeshUpdate();


        /*
        if(!_meshCrew->sampler().useComputedMetric())
            return;

        struct EdgeStats{
            vector<double> lengths;
            double minLen = 1.0/0.0;
            double maxLen = -1.0/0.0;
            double meanLen = 0.0;
        } euclide, riemann;

        MetricFreeMeasurer freeMeasurer;
        MetricWiseMeasurer wiseMeasurer;

        int vertCount = _mesh->verts.size();
        for(int v=0; v < vertCount; ++v)
        {
            const MeshTopo& topo = _mesh->topos[v];
            for(const MeshNeigVert& n : topo.neighborVerts)
            {
                if(v < n.v)
                {
                    double eucLen = freeMeasurer.riemannianDistance(
                        _meshCrew->sampler(),
                        _mesh->verts[v].p,
                        _mesh->verts[n.v].p,
                        _mesh->verts[v].c);

                    euclide.lengths.push_back(eucLen);
                    euclide.minLen = glm::min(euclide.minLen, eucLen);
                    euclide.maxLen = glm::max(euclide.maxLen, eucLen);
                    euclide.meanLen += eucLen;


                    double metLen = wiseMeasurer.riemannianDistance(
                        _meshCrew->sampler(),
                        _mesh->verts[v].p,
                        _mesh->verts[n.v].p,
                        _mesh->verts[v].c);

                    riemann.lengths.push_back(metLen);
                    riemann.minLen = glm::min(riemann.minLen, metLen);
                    riemann.maxLen = glm::max(riemann.maxLen, metLen);
                    riemann.meanLen += metLen;
                }
            }
        }

        getLog().postMessage(new Message('D', false,
            "Eucl stats : (min,\tmax,\tmean)", "GpuMesh"));
        getLog().postMessage(new Message('D', false,
            "\t" + to_string(euclide.minLen) +
            "\t" + to_string(euclide.maxLen) +
            "\t" + to_string(euclide.meanLen / euclide.lengths.size()),
            "GpuMesh"));

        getLog().postMessage(new Message('D', false,
            "Riem stats : (min,\tmax,\tmean)", "GpuMesh"));
        getLog().postMessage(new Message('D', false,
            "\t" + to_string(riemann.minLen) +
            "\t" + to_string(riemann.maxLen) +
            "\t" + to_string(riemann.meanLen / riemann.lengths.size()),
            "GpuMesh"));
        */
    }
}

void GpuMeshCharacter::updateSampling()
{
    if(_meshCrew->initialized())
    {
        _meshCrew->sampler().setScaling(_metricScaling);
        _meshCrew->sampler().setAspectRatio(_metricAspectRatio);
        _meshCrew->sampler().setDiscretizationDepth(_metricDiscretizationDepth);

        _meshCrew->sampler().updateAnalyticalMetric(*_mesh);
        _meshCrew->sampler().updateComputedMetric(*_mesh, _computedMetricSmapler);

        if(_renderer.get() != nullptr)
            _renderer->notifyMeshUpdate();
    }
}

void GpuMeshCharacter::setupInstalledRenderer()
{
    if(_renderer.get() != nullptr)
    {
        _renderer->setup();

        // Setup view matrix
        refreshCamera();

        // Setup shadow matrix
        moveLight(_lightAzimuth, _lightAltitude, _lightDistance);

        // Set cut type
        _renderer->useCutType(_cutType);

        // Display backdrop
        _renderer->displayBackdrop(_displayBackdrop);

        // Setup cut plane position
        moveCutPlane(_cutAzimuth, _cutAltitude, _cutDistance);

        setElementVisibility(
            _tetVisibility,
            _pyrVisibility,
            _priVisibility,
            _hexVisibility);

        setQualityCullingBounds(_qualityCullingMin, _qualityCullingMax);

        // Setup viewport
        play().view()->camera3D()->registerObserver(*_renderer);
        play().view()->camera3D()->refresh();
    }
}

void GpuMeshCharacter::tearDownInstalledRenderer()
{
    if(_renderer.get() != nullptr)
    {
        play().view()->camera3D()->unregisterObserver(*_renderer);
        _renderer->tearDown();
    }
}

void GpuMeshCharacter::installRenderer(const std::shared_ptr<AbstractRenderer>& renderer)
{
    if(_meshCrew->initialized())
        tearDownInstalledRenderer();

    _renderer = renderer;

    if(_meshCrew->initialized())
        setupInstalledRenderer();
}

void GpuMeshCharacter::moveObliqueCamera()
{
    glm::vec3 from(0.0f, 0.0f, 0.0f);
    glm::vec3 to(0.0f, -1.0f, 0.0f);

    glm::mat4 viewMatrix = glm::lookAt(from, to, upVec);
    play().view()->camera3D()->updateView(viewMatrix);
}

void GpuMeshCharacter::moveSphereCamera(float azimuth, float altitude, float distance)
{
    const float PI = glm::pi<float>();
    _camAzimuth = glm::mod(azimuth, 2 * PI);
    _camAltitude = glm::clamp(altitude, -PI * 0.48f, PI * 0.48f);
    _camDistance = glm::clamp(distance, 0.05f, 10.0f);

    glm::vec3 from = glm::vec3(
            glm::rotate(glm::mat4(), _camAzimuth, glm::vec3(0, 0, 1)) *
            glm::rotate(glm::mat4(), _camAltitude, glm::vec3(0, 1, 0)) *
            glm::vec4(_camDistance, 0.0f, 0.0f, 1.0f));

    glm::mat4 viewMatrix = glm::lookAt(from, nullVec, upVec);
    play().view()->camera3D()->updateView(viewMatrix);

    glm::vec3 camPos(from);

    _renderer->updateCamera(camPos);
}

void GpuMeshCharacter::moveFreeCamera()
{
    _renderer->updateCamera(_cameraManFree->position());
}

void GpuMeshCharacter::moveLight(float azimuth, float altitude, float distance)
{
    const float PI = glm::pi<float>();
    _lightAzimuth = glm::mod(azimuth, 2 *PI );
    _lightAltitude = glm::clamp(altitude, -PI / 2, PI / 2);
    _lightDistance = glm::clamp(distance, 1.0f, 10.0f);

    glm::vec3 from = -glm::vec3(
            glm::rotate(glm::mat4(), _lightAzimuth, glm::vec3(0, 0, 1)) *
            glm::rotate(glm::mat4(), _lightAltitude, glm::vec3(0, 1, 0)) *
            glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));

    glm::mat4 view = glm::lookAt(nullVec, from , upVec);

    _renderer->updateLight(view, from);
}

void GpuMeshCharacter::moveCutPlane(double azimuth, double altitude, double distance)
{
    const double PI = glm::pi<double>();
    _cutAzimuth = glm::mod(azimuth, 2 * PI);
    _cutAltitude = glm::clamp(altitude, -PI / 2, PI / 2);
    _cutDistance = glm::clamp(distance, -2.0, 2.0);

    glm::dvec4 cutPlaneEq =
            glm::rotate(glm::dmat4(), _cutAzimuth,  glm::dvec3(0, 0, 1)) *
            glm::rotate(glm::dmat4(), _cutAltitude, glm::dvec3(0, 1, 0)) *
            glm::dvec4(1.0, 0.0, 0.0, 1.0);
    cutPlaneEq.w = _cutDistance;

    _renderer->updateCutPlane(cutPlaneEq);
}
