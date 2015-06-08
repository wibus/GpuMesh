#include "GpuMeshCharacter.h"

#include <sstream>
#include <chrono>

#include <GLM/glm.hpp>
#include <GLM/gtc/random.hpp>
#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Misc/Log.h>

#include <Scaena/Play/Play.h>
#include <Scaena/Play/View.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

using namespace std;
using namespace cellar;
using namespace scaena;



GpuMeshCharacter::GpuMeshCharacter() :
    Character("GpuMeshChracter"),
    _vao(0),
    _vbo(0),
    _ibo(0),
    _azimuth(0),
    _altitude(0),
    _distance(10),
    _internalVertices(25000),
    _useGpuPipeline(false),
    _processFinished(false),
    _stepId(0)
{
}

void GpuMeshCharacter::enterStage()
{
    setupShaders();
    resetCpuPipeline();
    resetGpuPipeline();

    _processFinished = false;
    _stepId = 0;
}

void GpuMeshCharacter::beginStep(const StageTime &time)
{
    if(!_processFinished)
    {
        auto startTime = chrono::high_resolution_clock::now();

        if(_useGpuPipeline)
        {
            processGpuPipeline();
        }
        else
        {
            processCpuPipeline();
        }

        auto endTime = chrono::high_resolution_clock::now();
        auto dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);

        stringstream ss;
        ss << "Step took " << dt.count() / 1000.0 << "ms to execute";
        getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
    }


    std::shared_ptr<SynchronousMouse> mouse = play().synchronousMouse();
    if(mouse->buttonIsPressed(EMouseButton::LEFT))
    {
        moveCamera(_azimuth - mouse->displacement().x / 100.0f,
                   _altitude - mouse->displacement().y / 100.0f,
                   _distance);
    }
    else if(mouse->degreeDelta() != 0)
    {
        moveCamera(_azimuth, _altitude,
                   _distance + mouse->degreeDelta() / 20.0f);
    }
}

void GpuMeshCharacter::draw(const shared_ptr<View>& view, const StageTime& time)
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    _shader.pushProgram();
    glBindVertexArray(_vao);


    // Render tangle in wireframe
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    _shader.setVec3f("Color", glm::dvec3(0.2, 0.2, 0.2));
    glDrawElements(GL_TRIANGLES, _mesh.elemCount(), GL_UNSIGNED_INT, (GLvoid*) 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_CULL_FACE);

/*
    // Render vertices as points
    glPointSize(6.0f);
    _shader.setVec3f("Color", glm::dvec3(0.2, 0.65, 0.2));
    _shader.setVec3f("Offset", glm::dvec3(0, 0, -0.01));
    glDrawArrays(GL_POINTS, 0, _mesh.vertCount());
    _shader.setVec3f("Offset", glm::dvec3(0, 0, 0));
*/

    glBindVertexArray(0);
    _shader.popProgram();
}

void GpuMeshCharacter::exitStage()
{
}

void GpuMeshCharacter::printStep(int step, const std::string& stepName)
{
    stringstream ss;
    ss << "Step " << step << ": Executing " << stepName;
    getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
}

void GpuMeshCharacter::moveCamera(double azimuth, double altitude, double distance)
{
    _azimuth = glm::mod(azimuth, 2.0 * glm::pi<double>());
    _altitude = glm::clamp(altitude, -glm::pi<double>() * 0.48, glm::pi<double>() * 0.48);
    _distance = glm::clamp(distance, 0.05, 20.0);

    glm::vec4 from = glm::rotate(glm::dmat4(), _azimuth, glm::dvec3(0, 0, 1)) *
                     glm::rotate(glm::dmat4(), _altitude, glm::dvec3(0, 1, 0)) *
                     glm::vec4(_distance, 0.0f, 0.0f, 1.0f);

    _viewMatrix = glm::lookAt(glm::dvec3(from),
                              glm::dvec3(0, 0, 0),
                              glm::dvec3(0, 0, 1));

    _shader.pushProgram();
    _shader.setMat4f("Vmat", _viewMatrix);
    _shader.popProgram();
}

void GpuMeshCharacter::setupShaders()
{
    glm::dvec2 viewport = glm::dvec2(play().view()->viewport());
    _projection = glm::perspectiveFov(glm::pi<double>() / 8, viewport.x, viewport.y, 0.1, 20.0);
    _viewMatrix = glm::lookAt(glm::dvec3(_distance, 0, 0), glm::dvec3(), glm::dvec3(0, 0, 1));

    _shader.addShader(GL_VERTEX_SHADER, ":/shaders/Boundary.vert");
    _shader.addShader(GL_FRAGMENT_SHADER, ":/shaders/Boundary.frag");
    _shader.link();
    _shader.pushProgram();
    _shader.setMat4f("Vmat", _viewMatrix);
    _shader.setMat4f("Pmat", _projection);
    _shader.setVec3f("Color", glm::dvec3(1, 1, 1));
    _shader.setVec3f("Offset", glm::dvec3(0, 0, 0));
    _shader.popProgram();
}

void GpuMeshCharacter::updateBuffers()
{
    vector<unsigned int> indices;
    vector<glm::dvec3> vertices;
    _mesh.compileArrayBuffers(indices, vertices);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    GLuint verticesSize = vertices.size() * sizeof(decltype(vertices.front()));
    glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    GLuint indicesSize = indices.size() * sizeof(unsigned int);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesSize, indices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


// CPU Pipeline
void GpuMeshCharacter::resetCpuPipeline()
{
    // Delete old buffers
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;
    glDeleteBuffers(1, &_vbo);
    _vbo = 0;
    glDeleteBuffers(1, &_ibo);


    // Generate new buffers
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GpuMeshCharacter::processCpuPipeline()
{
    switch(_stepId)
    {
    case 0:
        printStep(_stepId, "Generation of boundary surfaces");
        genBoundaryMeshesCpu();
        ++_stepId;
        break;

    case 1:
        printStep(_stepId, "Triangulation of internal domain");
        triangulateDomainCpu();
        ++_stepId;
        break;

    case 2:
        printStep(_stepId, "Smoothing of the internal domain");
        smoothMeshCpu();

        _processFinished = true;
        ++_stepId;
        break;

    default:
        _processFinished = true;
        getLog().postMessage(new Message(
            'E', false, "Invalid step", "GpuMeshCharacter"));
    }
}

void GpuMeshCharacter::genBoundaryMeshesCpu()
{
    vector<glm::dvec3> vertices;
    vertices.push_back(glm::dvec3(-1.0, -1.0,  1.0));
    vertices.push_back(glm::dvec3( 1.0, -1.0,  1.0));
    vertices.push_back(glm::dvec3(-1.0,  1.0,  1.0));
    vertices.push_back(glm::dvec3( 1.0,  1.0,  1.0));
    vertices.push_back(glm::dvec3(-1.0, -1.0, -1.0));
    vertices.push_back(glm::dvec3( 1.0, -1.0, -1.0));
    vertices.push_back(glm::dvec3(-1.0,  1.0, -1.0));
    vertices.push_back(glm::dvec3( 1.0,  1.0, -1.0));

    std::vector<Tetrahedron> tetrahedron;
    tetrahedron.push_back(Tetrahedron(0, 1, 2, 4));
    tetrahedron.push_back(Tetrahedron(5, 4, 7, 1));
    tetrahedron.push_back(Tetrahedron(3, 1, 7, 2));
    tetrahedron.push_back(Tetrahedron(6, 2, 7, 4));
    tetrahedron.push_back(Tetrahedron(4, 1, 2, 7));

    _mesh.initialize(vertices, tetrahedron);
    updateBuffers();
}

void GpuMeshCharacter::triangulateDomainCpu()
{
    glm::dvec3 min = glm::dvec3(-1);
    glm::dvec3 max = glm::dvec3( 1);
    std::vector<glm::dvec3> vertices(_internalVertices);
    for(int i=0; i<_internalVertices; ++i)
        vertices[i] = glm::linearRand(min, max);

    _mesh.insertVertices(vertices);
    //updateBuffers();
}

void GpuMeshCharacter::smoothMeshCpu()
{
    exit(0);
}


// GPU Pipeline
void GpuMeshCharacter::resetGpuPipeline()
{

}

void GpuMeshCharacter::processGpuPipeline()
{
    processCpuPipeline();
}
