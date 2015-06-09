#include "GpuMeshCharacter.h"

#include <sstream>
#include <chrono>
#include <iostream>

#include <GLM/glm.hpp>
#include <GLM/gtc/random.hpp>
#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Misc/Log.h>

#include <Scaena/Play/Play.h>
#include <Scaena/Play/View.h>
#include <Scaena/StageManagement/Event/SynchronousKeyboard.h>
#include <Scaena/StageManagement/Event/SynchronousMouse.h>

using namespace std;
using namespace cellar;
using namespace scaena;



GpuMeshCharacter::GpuMeshCharacter() :
    Character("GpuMeshChracter"),
    _buffElemCount(0),
    _vao(0),
    _vbo(0),
    _nbo(0),
    _ebo(0),
    _qbo(0),
    _camAzimuth(0),
    _camAltitude(0),
    _camDistance(6),
    _cutAzimuth(0),
    _cutAltitude(0),
    _cutDistance(0),
    _internalVertices(10000),
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
    std::shared_ptr<SynchronousKeyboard> keyboard = play().synchronousKeyboard();
    std::shared_ptr<SynchronousMouse> mouse = play().synchronousMouse();

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
    else
    {
        if(keyboard->isAsciiPressed('s'))
        {
            _processFinished = false;
            _stepId = 2;
        }
    }


    if(!keyboard->isNonAsciiPressed(ENonAscii::SHIFT))
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
            moveCamera(_camAzimuth, _camAltitude,
                       _camDistance + mouse->degreeDelta() / 80.0f);
        }

    }
    else
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
            moveCutPlane(_cutAzimuth, _cutAltitude,
                         _cutDistance + mouse->degreeDelta() / 800.0f);
        }
    }

}

void GpuMeshCharacter::draw(const shared_ptr<View>&, const StageTime&)
{
    glClearColor(0.93, 0.95, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    _shader.pushProgram();
    glBindVertexArray(_vao);
    glDrawArrays(GL_TRIANGLES, 0, _buffElemCount);
    glBindVertexArray(0);
    _shader.popProgram();
}

void GpuMeshCharacter::exitStage()
{
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
    double a = 20.0;

    vector<glm::dvec3> vertices;
    vertices.push_back(glm::dvec3(-a, -a,  a));
    vertices.push_back(glm::dvec3( a, -a,  a));
    vertices.push_back(glm::dvec3(-a,  a,  a));
    vertices.push_back(glm::dvec3( a,  a,  a));
    vertices.push_back(glm::dvec3(-a, -a, -a));
    vertices.push_back(glm::dvec3( a, -a, -a));
    vertices.push_back(glm::dvec3(-a,  a, -a));
    vertices.push_back(glm::dvec3( a,  a, -a));

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
    double sphereRadius = 1.0;
    glm::dvec3 cMin(-sphereRadius);
    glm::dvec3 cMax( sphereRadius);
    std::vector<glm::dvec3> vertices;


    //* Box distribution
    vertices.resize(_internalVertices);

    for(int iv=0; iv<_internalVertices; ++iv)
        vertices[iv] = glm::linearRand(cMin, cMax);

    _mesh.insertVertices(vertices);
    //*/

    /* Sphere distribution
    vertices.resize(_internalVertices);

    for(int iv=0; iv<_internalVertices; ++iv)
        vertices[iv] = glm::ballRand(sphereRadius * 1.41);

    _mesh.insertVertices(vertices);
    //*/


    updateBuffers();
}

void GpuMeshCharacter::smoothMeshCpu()
{
    std::vector<std::vector<int>> adjacency;
    _mesh.compileAdjacencyLists(adjacency);

    const glm::dvec3 MOVE_FACTOR(0.5);
    const double SMOOTH_AMELIORATION_THRESHOLD = 0.001;
    double dQuality = 1.0;
    int smoothPass = 0;

    while(dQuality > SMOOTH_AMELIORATION_THRESHOLD)
    {
        cout << "Smooth pass number" << smoothPass << endl;
        cout << "Input mesh quality mean: " << _mesh.qualityMean << endl;
        cout << "Input mesh quality std dev: " << _mesh.qualityVar << endl;
        double lastQualityMean = _mesh.qualityMean;

        int vertCount = _mesh.vertCount();
        int firstVert = _mesh.externalVertCount;
        for(int v = firstVert; v < vertCount; ++v)
        {
            if(_mesh.vert[v].isBoundary)
                continue;

            double weightSum = 0.0;
            glm::dvec3 barycenter;

            glm::dvec3& vertPos = _mesh.vert[v].p;
            for(auto& n : adjacency[v])
            {
                const glm::dvec3& neighborPos = _mesh.vert[n].p;
                glm::dvec3 dist = vertPos - neighborPos;
                double weight = glm::log(glm::dot(dist, dist) + 1);

                barycenter = (barycenter * weightSum + neighborPos * weight)
                              / (weightSum + weight);
                weightSum += weight;
            }

            vertPos = glm::mix(vertPos, barycenter, MOVE_FACTOR);
        }

        updateBuffers();

        dQuality = _mesh.qualityMean - lastQualityMean;
        ++smoothPass;
    }

    cout << "#Smoothing finished" << endl;
    cout << "Final mesh quality mean: " << _mesh.qualityMean << endl;
    cout << "Final mesh quality std dev: " << _mesh.qualityVar << endl << endl;
}


// GPU Pipeline
void GpuMeshCharacter::resetGpuPipeline()
{

}

void GpuMeshCharacter::processGpuPipeline()
{
    processCpuPipeline();
}


// Common management
void GpuMeshCharacter::printStep(int step, const std::string& stepName)
{
    stringstream ss;
    ss << "Step " << step << ": Executing " << stepName;
    getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
}

void GpuMeshCharacter::moveCamera(double azimuth, double altitude, double distance)
{
    _camAzimuth = glm::mod(azimuth, 2.0 * glm::pi<double>());
    _camAltitude = glm::clamp(altitude, -glm::pi<double>() * 0.48, glm::pi<double>() * 0.48);
    _camDistance = glm::clamp(distance, 0.05, 7.0);

    glm::vec4 from = glm::rotate(glm::dmat4(), _camAzimuth, glm::dvec3(0, 0, 1)) *
                     glm::rotate(glm::dmat4(), _camAltitude, glm::dvec3(0, 1, 0)) *
                     glm::vec4(_camDistance, 0.0f, 0.0f, 1.0f);

    _viewMatrix = glm::lookAt(glm::dvec3(from),
                              glm::dvec3(0, 0, 0),
                              glm::dvec3(0, 0, 1));

    _shader.pushProgram();
    _shader.setMat4f("PVmat", _projection * _viewMatrix);
    _shader.setVec3f("CameraPosition", glm::vec3(from));
    _shader.popProgram();
}

void GpuMeshCharacter::moveCutPlane(double azimuth, double altitude, double distance)
{
    _cutAzimuth = glm::mod(azimuth, 2.0 * glm::pi<double>());
    _cutAltitude = glm::clamp(altitude, -glm::pi<double>(), glm::pi<double>());
    _cutDistance = glm::clamp(distance, -2.0, 2.0);

    glm::vec4 cutPlaneEq =
            glm::rotate(glm::dmat4(), _cutAzimuth, glm::dvec3(0, 0, 1)) *
            glm::rotate(glm::dmat4(), _cutAltitude, glm::dvec3(0, 1, 0)) *
            glm::vec4(1.0, 0.0f, 0.0f, 1.0f);
    cutPlaneEq.w = _cutDistance;

    _shader.pushProgram();
    _shader.setVec4f("CutPlaneEq", cutPlaneEq);
    _shader.popProgram();
}

void GpuMeshCharacter::setupShaders()
{
    // Compile shader
    _shader.addShader(GL_VERTEX_SHADER, ":/shaders/Boundary.vert");
    _shader.addShader(GL_FRAGMENT_SHADER, ":/shaders/Boundary.frag");
    _shader.link();
    _shader.pushProgram();
    _shader.popProgram();

    // Setup projection
    glm::dvec2 viewport = glm::dvec2(play().view()->viewport());
    _projection = glm::perspectiveFov(
            glm::pi<double>() / 6,
            viewport.x,
            viewport.y,
            0.1,
            10.0);

    // Setup view matrix
    moveCamera(_camAzimuth, _camAltitude, _camDistance);

    // Setup cut plane
    moveCutPlane(_cutAzimuth, _cutAltitude, _cutDistance);
}

void GpuMeshCharacter::updateBuffers()
{
    vector<glm::dvec3> vertices;
    vector<glm::dvec3> normals;
    vector<glm::dvec3> edges;
    vector<double> qualities;

    _mesh.compileFacesAttributes(vertices, normals, edges, qualities);
    _buffElemCount = vertices.size();

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    GLuint verticesSize = vertices.size() * sizeof(decltype(vertices.front()));
    glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, _nbo);
    GLuint normalsSize = normals.size() * sizeof(decltype(normals.front()));
    glBufferData(GL_ARRAY_BUFFER, normalsSize, normals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, _ebo);
    GLuint edgesSize = edges.size() * sizeof(decltype(edges.front()));
    glBufferData(GL_ARRAY_BUFFER, edgesSize, edges.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    GLuint qualitiesSize = qualities.size() * sizeof(decltype(qualities.front()));
    glBufferData(GL_ARRAY_BUFFER, qualitiesSize, qualities.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


// CPU Pipeline
void GpuMeshCharacter::resetCpuPipeline()
{
    // Delete old buffers
    glDeleteVertexArrays(1, &_vao);
    _vao = 0;

    glDeleteBuffers(1, &_vbo);
    _vbo = 0;

    glDeleteBuffers(1, &_nbo);
    _nbo = 0;

    glDeleteBuffers(1, &_ebo);
    _ebo = 0;

    glDeleteBuffers(1, &_qbo);
    _qbo = 0;

    _buffElemCount = 0;


    // Generate new buffers
    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_nbo);
    glBindBuffer(GL_ARRAY_BUFFER, _nbo);
    glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(1);

    glGenBuffers(1, &_ebo);
    glBindBuffer(GL_ARRAY_BUFFER, _ebo);
    glVertexAttribPointer(2, 3, GL_DOUBLE, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(2);

    glGenBuffers(1, &_qbo);
    glBindBuffer(GL_ARRAY_BUFFER, _qbo);
    glVertexAttribPointer(3, 1, GL_DOUBLE, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(3);
}
