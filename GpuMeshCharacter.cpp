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
#include "Evaluators/SolidAngleEvaluator.h"
#include "Evaluators/VolumeEdgeEvaluator.h"
#include "Meshers/CpuDelaunayMesher.h"
#include "Meshers/CpuParametricMesher.h"
#include "Renderers/ScaffoldRenderer.h"
#include "Renderers/SurfacicRenderer.h"
#include "Smoothers/SpringLaplaceSmoother.h"
#include "Smoothers/QualityLaplaceSmoother.h"

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
    _lightAzimuth(-glm::pi<float>() * 3.5 / 8.0),
    _lightAltitude(-glm::pi<float>() * 2.0 / 4.0),
    _lightDistance(1.0),
    _cutAzimuth(0),
    _cutAltitude(-glm::pi<float>() / 2.0),
    _cutDistance(0),
    _mesh(new GpuMesh())
{
    _availableMeshers = decltype(_availableMeshers){
        {string("Delaunay"),   shared_ptr<AbstractMesher>(new CpuDelaunayMesher())},
        {string("Parametric"), shared_ptr<AbstractMesher>(new CpuParametricMesher())},
    };

    _availableEvaluators = decltype(_availableEvaluators){
        {string("Insphere Edge"), shared_ptr<AbstractEvaluator>(new InsphereEdgeEvaluator())},
        {string("Solid Angle"),   shared_ptr<AbstractEvaluator>(new SolidAngleEvaluator())},
        {string("Volume Edge"),   shared_ptr<AbstractEvaluator>(new VolumeEdgeEvaluator())},
    };

    _availableSmoothers = decltype(_availableSmoothers){
        {string("Spring Laplace"), shared_ptr<AbstractSmoother>(new SpringLaplaceSmoother())},
        {string("Quality Laplace"), shared_ptr<AbstractSmoother>(new QualityLaplaceSmoother())},
    };

    _availableRenderers = decltype(_availableRenderers){
        {string("Scaffold"), shared_ptr<AbstractRenderer>(new ScaffoldRenderer())},
        {string("Surfacic"), shared_ptr<AbstractRenderer>(new SurfacicRenderer())},
    };
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

    _renderer->handleInputs(*keyboard, *mouse);
}

void GpuMeshCharacter::draw(const shared_ptr<View>&, const StageTime& time)
{
    _fps->setText("UPS: " + to_string(time.framesPerSecond()));
    _renderer->display(*_mesh, *_visualEvaluator);
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

std::vector<std::string> GpuMeshCharacter::availableMeshers() const
{
    std::vector<std::string> keyVec;
    for(const auto& keyValue : _availableMeshers)
        keyVec.push_back(keyValue.first);
    return keyVec;
}

std::vector<std::string> GpuMeshCharacter::availableMeshModels(const string& mesherName) const
{
    auto it = _availableMeshers.find(mesherName);
    if(it != _availableMeshers.end())
    {
        return it->second->availableMeshModels();
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + mesherName + "' mesher", "GpuMeshCharacter"));
        return std::vector<std::string>();
    }
}

std::vector<std::string> GpuMeshCharacter::availableEvaluators() const
{
    std::vector<std::string> keyVec;
    for(const auto& keyValue : _availableEvaluators)
        keyVec.push_back(keyValue.first);
    return keyVec;
}

std::vector<std::string> GpuMeshCharacter::availableSmoothers() const
{
    std::vector<std::string> keyVec;
    for(const auto& keyValue : _availableSmoothers)
        keyVec.push_back(keyValue.first);
    return keyVec;
}

std::vector<std::string> GpuMeshCharacter::availableImplementations(const string& smootherName) const
{
    auto it = _availableSmoothers.find(smootherName);
    if(it != _availableSmoothers.end())
    {
        return it->second->availableImplementations();
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + smootherName + "' smoother", "GpuMeshCharacter"));
        return std::vector<std::string>();
    }
}

std::vector<std::string> GpuMeshCharacter::availableRenderers() const
{
    std::vector<std::string> keyVec;
    for(const auto& keyValue : _availableRenderers)
        keyVec.push_back(keyValue.first);
    return keyVec;
}

std::vector<std::string> GpuMeshCharacter::availableShadings() const
{
    return _renderer->availableShadings();
}

void GpuMeshCharacter::generateMesh(
        const std::string& mesherName,
        const std::string& modelName,
        size_t vertexCount)
{
    _mesh->clear();

    printStep("Mesh Generation: mesher=" + mesherName +
              ", model=" + modelName +
              ", vertex count=" + to_string(vertexCount));

    auto it = _availableMeshers.find(mesherName);
    if(it != _availableMeshers.end())
    {
        _availableMeshers[mesherName]->generateMesh(
                    *_mesh,
                    modelName,
                    vertexCount);

        _mesh->compileTopoly();
        _renderer->notifyMeshUpdate();
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + mesherName + "' mesher", "GpuMeshCharacter"));
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
    printStep("Mesh Smoothing: smoother=" + smootherName +
              ", quality measure=" + evaluatorName +
              ", implementation=" + implementationName);

    auto it = _availableSmoothers.find(smootherName);
    if(it != _availableSmoothers.end())
    {
        auto evaluator = _availableEvaluators.find(evaluatorName);
        if(evaluator != _availableEvaluators.end())
        {
            _availableSmoothers[smootherName]->smoothMesh(
                        *_mesh,
                        *evaluator->second,
                        implementationName,
                        minIterationCount,
                        moveFactor,
                        gainThreshold);

            _renderer->notifyMeshUpdate();
        }
        else
        {
            getLog().postMessage(new Message('E', false,
                "Failed to find '" + evaluatorName + "' evaluator", "GpuMeshCharacter"));
        }
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + smootherName + "' smoother", "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::useRenderer(const std::string& rendererName)
{
    auto it = _availableRenderers.find(rendererName);
    if(it != _availableRenderers.end())
    {
        installRenderer(it->second);
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + rendererName + "' renderer", "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::useShading(const std::string& shadingName)
{
    _renderer->useShading(shadingName);
}

void GpuMeshCharacter::displayQuality(const std::string& evaluatorName)
{
    auto it = _availableEvaluators.find(evaluatorName);
    if(it != _availableEvaluators.end())
    {
        _visualEvaluator = it->second;
        _renderer->notifyMeshUpdate();
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + evaluatorName + "' evaluator", "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::useVirtualCutPlane(bool use)
{
    _renderer->useVirtualCutPlane(use);
}

void GpuMeshCharacter::printStep(const std::string& stepDescription)
{
    getLog().postMessage(new Message('I', false, stepDescription, "GpuMeshCharacter"));
}

void GpuMeshCharacter::setupInstalledRenderer()
{
    if(_renderer.get() != nullptr)
    {
        _renderer->setup();

        // Setup view matrix
        moveCamera(_camAzimuth, _camAltitude, _camDistance);

        // Setup shadow matrix
        moveLight(_lightAzimuth, _lightAltitude, _lightDistance);

        // Setup cut plane
        moveCutPlane(_cutAzimuth, _cutAltitude, _cutDistance);

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
