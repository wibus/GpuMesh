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
#include "Evaluators/InsphereEvaluator.h"
#include "Evaluators/SolidAngleEvaluator.h"
#include "Meshers/CpuDelaunayMesher.h"
#include "Meshers/CpuParametricMesher.h"
#include "Renderers/MidEndRenderer.h"
#include "Renderers/ScientificRenderer.h"
#include "Smoothers/QualityLaplaceSmoother.h"

using namespace std;
using namespace cellar;
using namespace prop2;
using namespace scaena;



const glm::vec3 GpuMeshCharacter::nullVec = glm::vec3(0, 0, 0);
const glm::vec3 GpuMeshCharacter::upVec = glm::vec3(0, 0, 1);

GpuMeshCharacter::GpuMeshCharacter() :
    Character("GpuMeshChracter"),
    _camAzimuth(-glm::pi<float>() / 2.0),
    _camAltitude(-glm::pi<float>() / 2.0),
    _camDistance(3.8),
    _lightAzimuth(-glm::pi<float>() * 3.5 / 8.0),
    _lightAltitude(-glm::pi<float>() * 2.0 / 4.0),
    _lightDistance(1.0),
    _cutAzimuth(0),
    _cutAltitude(-glm::pi<float>() / 2.0),
    _cutDistance(0),
    _mesh(new GpuMesh()),
    _mesher(new CpuParametricMesher(1e6)),
    _smoother(new QualityLaplaceSmoother(200, 0.3, 0.0)),
    _evaluator(new SolidAngleEvaluator()),
    _renderer(nullptr),
    _rendererId(-1)
{
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

    installNextRenderer();

    resetPipeline();
}

void GpuMeshCharacter::beginStep(const StageTime &time)
{
    _ups->setText("UPS: " + to_string(time.framesPerSecond()));

    if(!_processFinished)
    {
        auto startTime = chrono::high_resolution_clock::now();

        processPipeline();

        auto endTime = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

        stringstream ss;
        ss << "Step took " << dt.count() / 1000.0 << "ms to execute";
        getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
    }    


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
    _renderer->display(*_mesh, *_evaluator);
}

void GpuMeshCharacter::exitStage()
{
    _renderer->tearDown();
}

bool GpuMeshCharacter::keyPressEvent(const scaena::KeyboardEvent &event)
{
    if(_processFinished)
    {
        if(event.getAscii() == 'A')
        {
            scheduleCpuSmoothing();
        }
        else if(event.getAscii() == 'S')
        {
            scheduleGpuSmoothing();
        }
        else if(event.getAscii() == 'Z')
        {
            installNextRenderer();

            const char* rep = _rendererId == 0 ? "Wireframe" : "Surfacic";
            cout << "Switching renderer to " << rep << " mode" << endl;
        }

        _renderer->handleKeyPress(event);
    }
}

void GpuMeshCharacter::resetPipeline()
{
    _stepId = 0;
    _processFinished = false;

    _mesh->clear();
}

void GpuMeshCharacter::processPipeline()
{
    switch(_stepId)
    {
    case 0:
        printStep(_stepId, "Triangulating internal domain");
        _mesher->triangulateDomain(*_mesh);

        printStep(_stepId, "Generating vertex adjacency lists");
        _mesh->compileTopoly();

        _renderer->notifyMeshUpdate();
        _processFinished = true;
        ++_stepId;
        break;

    case 1:
        printStep(_stepId, string("Smoothing internal domain ")
                    + (_gpuSmoothing ? "(GPU)" : "CPU") );

        if(_gpuSmoothing)
            _smoother->smoothGpuMesh(*_mesh, *_evaluator);
        else
            _smoother->smoothCpuMesh(*_mesh, *_evaluator);


        _renderer->notifyMeshUpdate();
        _processFinished = true;
        break;

    default:
        _processFinished = true;
        getLog().postMessage(new Message(
            'E', false, "Invalid step", "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::scheduleCpuSmoothing()
{
    if(_processFinished)
    {
        _processFinished = false;
        _gpuSmoothing = false;
    }
}

void GpuMeshCharacter::scheduleGpuSmoothing()
{
    if(_processFinished)
    {
        _processFinished = false;
        _gpuSmoothing = true;
    }
}

void GpuMeshCharacter::printStep(int step, const std::string& stepName)
{
    stringstream ss;
    ss << "Step " << step << ": " << stepName;
    getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
}

int GpuMeshCharacter::installNextRenderer()
{
    _rendererId = (_rendererId + 1) % 2;
    switch(_rendererId)
    {
    case 0 : installRenderer(new ScientificRenderer()); break;
    case 1 : installRenderer(new MidEndRenderer()); break;
    default : getLog().postMessage(new Message('W', false,
         "Invalid renderer ID: " + toString(_rendererId), "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::installRenderer(AbstractRenderer* renderer)
{
    if(_renderer.get() != nullptr)
    {
        play().view()->camera3D()->unregisterObserver(*_renderer);
        _renderer->tearDown();
    }

    _renderer.reset(renderer);
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
