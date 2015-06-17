#include "AbstractMesher.h"

#include <chrono>
#include <sstream>
#include <iostream>

#include <GLM/glm.hpp>
#include <GLM/gtc/random.hpp>

#include <CellarWorkbench/Misc/Log.h>

using namespace std;
using namespace cellar;


AbstractMesher::AbstractMesher(Mesh& mesh, unsigned int vertCount) :
    _mesh(mesh),
    _vertCount(vertCount),
    _processFinished(false),
    _stepId(0)
{

}

AbstractMesher::~AbstractMesher()
{

}

bool AbstractMesher::processFinished() const
{
    return _processFinished;
}

void AbstractMesher::resetPipeline()
{
    _stepId = 0;
    _processFinished = false;
}

void AbstractMesher::processPipeline()
{
    switch(_stepId)
    {
    case 0:
        printStep(_stepId, "Generation of boundary surfaces");
        genBoundaryMeshes();
        ++_stepId;
        break;

    case 1:
        printStep(_stepId, "Triangulation of internal domain");
        triangulateDomain();

        _processFinished = true;
        ++_stepId;
        break;

    case 2:
        printStep(_stepId, "Computing adjacency lists");
        computeAdjacency();
        ++_stepId;
        break;

    case 3:
        printStep(_stepId, "Smoothing of the internal domain");
        smoothMesh();

        _processFinished = true;
        break;

    default:
        _processFinished = true;
        getLog().postMessage(new Message(
            'E', false, "Invalid step", "GpuMeshCharacter"));
    }
}


void AbstractMesher::scheduleSmoothing()
{
    if(_processFinished)
    {
        _stepId = 2;
        _processFinished = false;
    }
}

void AbstractMesher::genBoundaryMeshes()
{
    const double a = 20.0;

    std::vector<glm::dvec3> vertices;
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
}

void AbstractMesher::triangulateDomain()
{
    chrono::high_resolution_clock::time_point startTime, endTime;
    chrono::microseconds dt;

    double sphereRadius = 1.0;
    glm::dvec3 cMin(-sphereRadius);
    glm::dvec3 cMax( sphereRadius);
    std::vector<glm::dvec3> vertices;


    //* Box distribution
    vertices.resize(_vertCount);

    int surfCount = glm::sqrt(_vertCount) * 10;
    int padding = glm::pow(_vertCount, 1/3.0);
    for(int iv=0; iv < surfCount; ++iv)
    {
        glm::dvec3 val = glm::linearRand(cMin, cMax);
        val[iv%3] = glm::mix(-1.0, 1.0, (double)(iv%2));
        vertices[iv] = val;
    }

    for(int iv=surfCount; iv<_vertCount; ++iv)
        vertices[iv] = glm::linearRand(cMin, cMax) * (1.0 - 1.0 / padding);

    startTime = chrono::high_resolution_clock::now();
    _mesh.insertVertices(vertices);
    endTime = chrono::high_resolution_clock::now();

    dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Total meshing time = " << dt.count() / 1000.0 << "ms" << endl;
    //*/

    /* Sphere distribution
    vertices.resize(_vertCount);

    for(int iv=0; iv<_vertCount; ++iv)
        vertices[iv] = glm::ballRand(sphereRadius * 1.41);

    startTime = chrono::high_resolution_clock::now();
    _mesh.insertVertices(vertices);
    endTime = chrono::high_resolution_clock::now();

    dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Total meshing time = " << dt.count() / 1000.0 << "ms" << endl;
    //*/


    /* Paraboloid cap
    int stackCount = glm::pow(_vertCount/2.3, 1/3.0);
    glm::dvec3 minPerturb(-0.1 / stackCount);
    glm::dvec3 maxPerturb(0.1 / stackCount);
    for(int s=0; s <= stackCount; ++s)
    {
        double sProg = s / (double) stackCount;
        double height = sProg;
        vertices.push_back(glm::dvec3(0, 0, height));

        for(int r=1; r <= s; ++r)
        {
            double rProg = r / (double) s;
            double ringRadius = r / (double) stackCount;
            double ringHeight = height * (1.0 - rProg*rProg);
            double ringVertCount = stackCount * glm::sqrt(1.0 + r);
            for(int v=0; v < ringVertCount; ++v)
            {
                double vProg = v / (double) ringVertCount;
                double vAngle = glm::pi<double>() * vProg * 2;
                vertices.push_back(glm::dvec3(
                    glm::cos(vAngle) * ringRadius,
                    glm::sin(vAngle) * ringRadius * 0.75,
                    ringHeight)
                                   +
                    glm::linearRand(minPerturb, maxPerturb)
                     * (1.1 - rProg) * (1.1 - sProg));
            }
        }
    }

    startTime = chrono::high_resolution_clock::now();
    _mesh.insertVertices(vertices);
    endTime = chrono::high_resolution_clock::now();

    dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Total meshing time = " << dt.count() / 1000.0 << "ms" << endl;
    //*/
}

void AbstractMesher::computeAdjacency()
{
    _mesh.compileAdjacencyLists();
}

void AbstractMesher::smoothMesh()
{

}

void AbstractMesher::printStep(int step, const std::string& stepName)
{
    stringstream ss;
    ss << "Step " << step << ": Executing " << stepName;
    getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
}
