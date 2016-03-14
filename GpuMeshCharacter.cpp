#include "GpuMeshCharacter.h"

#include <sstream>
#include <iostream>

#include <GLM/gtx/transform.hpp>

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
#include "Samplers/DummySampler.h"
#include "Samplers/UniformSampler.h"
#include "Samplers/KdTreeSampler.h"
#include "Samplers/LocalSampler.h"
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
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonDeserializer.h"
#include "Serialization/StlSerializer.h"
#include "Smoothers/VertexWise/SpringLaplaceSmoother.h"
#include "Smoothers/VertexWise/QualityLaplaceSmoother.h"
#include "Smoothers/VertexWise/LocalOptimisationSmoother.h"
#include "Smoothers/ElementWise/GetmeSmoother.h"

using namespace std;
using namespace cellar;
using namespace prop2;
using namespace scaena;


const std::string NO_METRIC_SAMPLING = "None";
const glm::vec3 GpuMeshCharacter::nullVec = glm::vec3(0, 0, 0);
const glm::vec3 GpuMeshCharacter::upVec = glm::vec3(0, 0, 1);

GpuMeshCharacter::GpuMeshCharacter() :
    Character("GpuMeshChracter"),
    _camAzimuth(-glm::pi<float>() / 2.0),
    _camAltitude(-glm::pi<float>() / 2.0),
    _camDistance(3.8),
    _cameraMan(ECameraMan::Sphere),
    _lightAzimuth(-glm::pi<float>() * 3.5 / 8.0),
    _lightAltitude(-glm::pi<float>() * 2.0 / 4.0),
    _lightDistance(1.0),
    _cutAzimuth(0),
    _cutAltitude(-glm::pi<double>() / 2.0),
    _cutDistance(1.0e-8),
    _tetVisibility(true),
    _priVisibility(true),
    _hexVisibility(true),
    _qualityCullingMin(-INFINITY),
    _qualityCullingMax(INFINITY),
    _samplingDensity(2),
    _displaySamplingMesh(false),
    _mesh(new GpuMesh()),
    _meshCrew(new MeshCrew()),
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

    _availableSamplers.setDefault("Local");
    _availableSamplers.setContent({
        {NO_METRIC_SAMPLING, shared_ptr<AbstractSampler>(new DummySampler())},
        {string("Analytic"), shared_ptr<AbstractSampler>(new AnalyticSampler())},
        {string("Uniform"),  shared_ptr<AbstractSampler>(new UniformSampler())},
        {string("Kd-Tree"),  shared_ptr<AbstractSampler>(new KdTreeSampler())},
        {string("Local"),    shared_ptr<AbstractSampler>(new LocalSampler())},
    });

    _availableEvaluators.setDefault("Metric Conformity");
    _availableEvaluators.setContent({
        {string("Mean Ratio"),        shared_ptr<AbstractEvaluator>(new MeanRatioEvaluator())},
        {string("Metric Conformity"), shared_ptr<AbstractEvaluator>(new MetricConformityEvaluator())},
    });

    _availableSmoothers.setDefault("Local Optimisation");
    _availableSmoothers.setContent({
        {string("Spring Laplace"),     shared_ptr<AbstractSmoother>(new SpringLaplaceSmoother())},
        {string("Quality Laplace"),    shared_ptr<AbstractSmoother>(new QualityLaplaceSmoother())},
        {string("Local Optimisation"), shared_ptr<AbstractSmoother>(new LocalOptimisationSmoother())},
        {string("GETMe"),              shared_ptr<AbstractSmoother>(new GetmeSmoother())},
    });

    _availableRenderers.setDefault("Surfacic");
    _availableRenderers.setContent({
        {string("Blind"),    shared_ptr<AbstractRenderer>(new BlindRenderer())},
        {string("Scaffold"), shared_ptr<AbstractRenderer>(new ScaffoldRenderer())},
        {string("Surfacic"), shared_ptr<AbstractRenderer>(new SurfacicRenderer())},
    });

    _availableCameraMen.setDefault("Sphere");
    _availableCameraMen.setContent({
        {string("Sphere"), ECameraMan::Sphere},
        {string("Free"),   ECameraMan::Free},
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
        {string("json"), shared_ptr<AbstractDeserializer>(new JsonDeserializer())},
    });
}

void GpuMeshCharacter::enterStage()
{
    _fps = play().propTeam2D()->createTextHud();
    _fps->setHandlePosition(glm::dvec2(5, 5));
    _fps->setHorizontalAnchor(EHorizontalAnchor::LEFT);
    _fps->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _fps->setHeight(16);

    _ups = play().propTeam2D()->createTextHud();
    _ups->setHandlePosition(glm::dvec2(5, 25));
    _ups->setHorizontalAnchor(EHorizontalAnchor::LEFT);
    _ups->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _ups->setHeight(16);

    _qualityTitle = play().propTeam2D()->createTextHud();
    _qualityTitle->setHandlePosition(glm::dvec2(-160, 300));
    _qualityTitle->setHorizontalAnchor(EHorizontalAnchor::RIGHT);
    _qualityTitle->setVerticalAnchor(EVerticalAnchor::BOTTOM);
    _qualityTitle->setHeight(20);
    _qualityTitle->setText("Element Quality");

    _qualityMax = play().propTeam2D()->createTextHud();
    _qualityMax->setHandlePosition(glm::dvec2(-130, 269));
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
    DummySampler verifSampler;
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

    setupInstalledRenderer();
    _meshCrew->initialize(*_mesh);
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
        if(_cameraMan == ECameraMan::Sphere)
        {
            if(mouse->buttonIsPressed(EMouseButton::LEFT))
            {
                moveCamera(_camAzimuth - mouse->displacement().x / 100.0f,
                           _camAltitude - mouse->displacement().y / 100.0f,
                           _camDistance);
            }
            else if(mouse->degreeDelta() != 0)
            {
                moveCamera(_camAzimuth,
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
                _renderer->updateCamera(_cameraManFree->camera()->viewMatrix(),
                                        _cameraManFree->position());
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
    _mesh->clear();

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
        updateSampling();
        updateMeshMeasures();
    }
}

void GpuMeshCharacter::clearMesh()
{
    _mesh->clear();
    _mesh->modelName = "";
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

void GpuMeshCharacter::loadMesh(
        const std::string& fileName)
{
    printStep("Loading mesh from " + fileName);

    string ext = fileExt(fileName);
    shared_ptr<AbstractDeserializer> deserializer;
    if(_availableDeserializers.select(ext, deserializer))
    {
        _mesh->clear();

        if(deserializer->deserialize(fileName, *_mesh))
        {
            _mesh->compileTopology();
            updateSampling();
            updateMeshMeasures();
        }
        else
        {
            getLog().postMessage(new Message('E', false,
                "An error occured while loading the mesh.", "GpuMeshCharacter"));
        }
    }
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
        evaluator->evaluateMesh(
            *_mesh,
            _meshCrew->sampler(),
            _meshCrew->measurer(),
            histogram,
            implementationName);

        getLog().postMessage(new Message('I', false,
            "Results "\
            ": min=" + to_string(histogram.minimumQuality()) +
            ", avg=" + to_string(histogram.averageQuality()), "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::benchmarkEvaluator(
        const std::string& evaluatorName,
        const map<string, int>& cycleCounts)
{
    printStep("Shape measure evaluation benchmark "\
              ": quality measure=" + evaluatorName);

    std::shared_ptr<AbstractEvaluator> evaluator;
    if(_availableEvaluators.select(evaluatorName, evaluator))
    {
        evaluator->benchmark(
            *_mesh,
            _meshCrew->sampler(),
            _meshCrew->measurer(),
            cycleCounts);
    }
}

void GpuMeshCharacter::smoothMesh(
        const std::string& smootherName,
        const std::string& implementationName,
        size_t minIterationCount,
        double moveFactor,
        double gainThreshold)
{
    printStep("Mesh Smoothing "\
              ": smoother=" + smootherName +
              ", implementation=" + implementationName);

    std::shared_ptr<AbstractSmoother> smoother;
    if(_availableSmoothers.select(smootherName, smoother))
    {
        smoother->smoothMesh(
            *_mesh,
            *_meshCrew,
            implementationName,
            minIterationCount,
            moveFactor,
            gainThreshold);

        updateSampling();
        updateMeshMeasures();
    }
}

OptimizationPlot GpuMeshCharacter::benchmarkSmoother(
        const std::string& smootherName,
        const map<string, bool>& activeImpls,
        size_t minIterationCount,
        double moveFactor,
        double gainThreshold)
{
    printStep("Smoothing benchmark "\
              ": smoother=" + smootherName);

    OptimizationPlot plot;
    _mesh->printPropperties(plot);
    plot.setMeshModelName(_mesh->modelName);
    plot.setSmoothingMethodName(smootherName);

    std::shared_ptr<AbstractSmoother> smoother;
    if(_availableSmoothers.select(smootherName, smoother))
    {
        smoother->benchmark(
            *_mesh,
            *_meshCrew,
            activeImpls,
            minIterationCount,
            moveFactor,
            gainThreshold,
            plot);

        updateSampling();
        updateMeshMeasures();
    }

    return plot;
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

void GpuMeshCharacter::useSamplingDensity(int density)
{
    _samplingDensity = density;
    updateSampling();
    updateMeshMeasures();
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

void GpuMeshCharacter::setElementVisibility(bool tet, bool pri, bool hex)
{
    _tetVisibility = tet;
    _priVisibility = pri;
    _hexVisibility = hex;

    if(_meshCrew->initialized())
    {
        _renderer->setElementVisibility(tet, pri, hex);
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

void GpuMeshCharacter::printStep(const std::string& stepDescription)
{
    getLog().postMessage(new Message('I', false, stepDescription, "GpuMeshCharacter"));
}

void GpuMeshCharacter::refreshCamera()
{
    if(_cameraMan == ECameraMan::Sphere)
        moveCamera(_camAzimuth, _camAltitude, _camDistance);
    else if(_cameraMan == ECameraMan::Free)
        _renderer->updateCamera(_cameraManFree->camera()->viewMatrix(),
                                _cameraManFree->position());
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
    }
}

void GpuMeshCharacter::updateSampling()
{
    if(_meshCrew->initialized())
    {
        _meshCrew->sampler().setReferenceMesh(
            *_mesh, _samplingDensity);

        if(_displaySamplingMesh)
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

        // Setup cut plane position
        moveCutPlane(_cutAzimuth, _cutAltitude, _cutDistance);

        setElementVisibility( _tetVisibility, _priVisibility, _hexVisibility);

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

void GpuMeshCharacter::moveCamera(float azimuth, float altitude, float distance)
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
    glm::vec3 camPos(from);

    _renderer->updateCamera(viewMatrix, camPos);
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
