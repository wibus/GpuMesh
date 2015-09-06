#include "GpuMeshCharacter.h"

#include <sstream>
#include <iostream>

#include <GLM/gtx/transform.hpp>

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
#include "Evaluators/InsphereEdgeEvaluator.h"
#include "Evaluators/MeanRatioEvaluator.h"
#include "Evaluators/SolidAngleEvaluator.h"
#include "Evaluators/VolumeEdgeEvaluator.h"
#include "Meshers/CpuDelaunayMesher.h"
#include "Meshers/CpuParametricMesher.h"
#include "Meshers/DebugMesher.h"
#include "Renderers/ScaffoldRenderer.h"
#include "Renderers/SurfacicRenderer.h"
#include "Renderers/QualityGradientPainter.h"
#include "Smoothers/VertexWise/SpringLaplaceSmoother.h"
#include "Smoothers/VertexWise/QualityLaplaceSmoother.h"
#include "Smoothers/VertexWise/LocalOptimisationSmoother.h"
#include "Smoothers/ElementWise/GetmeSmoother.h"

using namespace std;
using namespace cellar;
using namespace prop2;
using namespace scaena;



const glm::vec3 GpuMeshCharacter::nullVec = glm::vec3(0, 0, 0);
const glm::vec3 GpuMeshCharacter::upVec = glm::vec3(0, 0, 1);

GpuMeshCharacter::GpuMeshCharacter() :
    Character("GpuMeshChracter"),
    _isEntered(false),
    _camAzimuth(-glm::pi<float>() / 2.0),
    _camAltitude(-glm::pi<float>() / 2.0),
    _camDistance(3.8),
    _cameraMan(ECameraMan::Sphere),
    _lightAzimuth(-glm::pi<float>() * 3.5 / 8.0),
    _lightAltitude(-glm::pi<float>() * 2.0 / 4.0),
    _lightDistance(1.0),
    _cutAzimuth(0),
    _cutAltitude(-glm::pi<double>() / 2.0),
    _cutDistance(0),
    _tetVisibility(true),
    _priVisibility(true),
    _hexVisibility(true),
    _mesh(new GpuMesh()),
    _availableMeshers("Available Meshers"),
    _availableEvaluators("Available Evaluators"),
    _availableSmoothers("Available Smoothers"),
    _availableRenderers("Available Renderers"),
    _availableCameraMen("Available Camera Men"),
    _availableCutTypes("Available Cut Types")
{
    _availableMeshers.setDefault("Parametric");
    _availableMeshers.setContent({
        {string("Delaunay"),   shared_ptr<AbstractMesher>(new CpuDelaunayMesher())},
        {string("Parametric"), shared_ptr<AbstractMesher>(new CpuParametricMesher())},
        {string("Debug"),      shared_ptr<AbstractMesher>(new DebugMesher())},
    });

    _availableEvaluators.setDefault("Mean Ratio");
    _availableEvaluators.setContent({
        {string("Insphere Edge"), shared_ptr<AbstractEvaluator>(new InsphereEdgeEvaluator())},
        {string("Mean Ratio"),    shared_ptr<AbstractEvaluator>(new MeanRatioEvaluator())},
        {string("Solid Angle"),   shared_ptr<AbstractEvaluator>(new SolidAngleEvaluator())},
        {string("Volume Edge"),   shared_ptr<AbstractEvaluator>(new VolumeEdgeEvaluator())},
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
    for(auto evalName : _availableEvaluators.details().options)
    {
        std::shared_ptr<AbstractEvaluator> evaluator;
        if(_availableEvaluators.select(evalName, evaluator))
        {
            evaluator->assessMeasureValidy();
        }
    }

    _cameraManFree.reset(new CameraManFree(play().view()->camera3D(), false));
    _cameraManFree->setPosition(glm::vec3(0, 0, _camDistance));
    _cameraManFree->setOrientation( glm::pi<float>() * 0.5f,
                                   -glm::pi<float>() * 0.48f);

    setupInstalledRenderer();
    _isEntered = true;
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
    _renderer->display(*_mesh, *_evaluator);
}

void GpuMeshCharacter::exitStage()
{
    tearDownInstalledRenderer();
    _isEntered = false;
}

bool GpuMeshCharacter::keyPressEvent(const scaena::KeyboardEvent &event)
{
    _renderer->handleKeyPress(event);
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
        mesher->generateMesh( *_mesh, modelName, vertexCount);

        _mesh->compileTopoly();
        _renderer->notifyMeshUpdate();
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
        double minQuality, qualityMean;
        evaluator->evaluateMesh(
            *_mesh,
            minQuality,
            qualityMean,
            implementationName);

        getLog().postMessage(new Message('I', false,
            "Results "\
            ": min=" + to_string(minQuality) +
            ", mean=" + to_string(qualityMean), "GpuMeshCharacter"));
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
        evaluator->benchmark(*_mesh, cycleCounts);
    }
}

void GpuMeshCharacter::smoothMesh(
        const std::string& smootherName,
        const std::string& evaluatorName,
        const std::string& implementationName,
        size_t minIterationCount,
        double moveFactor,
        double gainThreshold)
{
    printStep("Mesh Smoothing "\
              ": smoother=" + smootherName +
              ", quality measure=" + evaluatorName +
              ", implementation=" + implementationName);

    std::shared_ptr<AbstractSmoother> smoother;
    if(_availableSmoothers.select(smootherName, smoother))
    {
        std::shared_ptr<AbstractEvaluator> evaluator;
        if(_availableEvaluators.select(evaluatorName, evaluator))
        {
            smoother->smoothMesh(
                *_mesh,
                *evaluator,
                implementationName,
                minIterationCount,
                moveFactor,
                gainThreshold);

            _renderer->notifyMeshUpdate();
        }
    }
}

OptimizationPlot GpuMeshCharacter::benchmarkSmoother(
        const std::string& smootherName,
        const string& evaluatorName,
        const map<string, bool>& activeImpls,
        size_t minIterationCount,
        double moveFactor,
        double gainThreshold)
{
    printStep("Smoothing benchmark "\
              ": smoother=" + smootherName +
              ", quality measure=" + evaluatorName);

    std::shared_ptr<AbstractSmoother> smoother;
    if(_availableSmoothers.select(smootherName, smoother))
    {
        std::shared_ptr<AbstractEvaluator> evaluator;
        if(_availableEvaluators.select(evaluatorName, evaluator))
        {
            return smoother->benchmark(
                *_mesh,
                *evaluator,
                activeImpls,
                minIterationCount,
                moveFactor,
                gainThreshold);
        }
    }

    return OptimizationPlot("Could not launch benchmarks");
}

void GpuMeshCharacter::useEvaluator(const std::string& evaluatorName)
{
    if(_availableEvaluators.select(evaluatorName, _evaluator))
    {
        if(_renderer.get() != nullptr)
            _renderer->notifyMeshUpdate();
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

    if(_isEntered)
    {
        refreshCamera();
    }
}

void GpuMeshCharacter::useCutType(const std::string& cutTypeName)
{
    _availableCutTypes.select(cutTypeName, _cutType);

    if(_isEntered)
    {
        _renderer->useCutType(_cutType);
    }
}

void GpuMeshCharacter::setElementVisibility(bool tet, bool pri, bool hex)
{
    if(_isEntered)
    {
        _tetVisibility = tet;
        _priVisibility = pri;
        _hexVisibility = hex;
        _renderer->setElementVisibility(tet, pri, hex);
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
    if(_isEntered)
        tearDownInstalledRenderer();

    _renderer = renderer;

    if(_isEntered)
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
