#include "AbstractMesher.h"

#include <sstream>
#include <iostream>

#include <CellarWorkbench/Misc/Log.h>

using namespace std;
using namespace cellar;


AbstractMesher::AbstractMesher(Mesh& mesh, unsigned int vertCount) :
    _mesh(mesh),
    _vertCount(vertCount),
    _stepId(0),
    _processFinished(false)
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
        printStep(_stepId, "Triangulating internal domain");
        triangulateDomain();

        _processFinished = true;
        ++_stepId;
        break;

    case 1:
        printStep(_stepId, "Smoothing internal domain");
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
        _stepId = 1;
        _processFinished = false;
    }
}

void AbstractMesher::printStep(int step, const std::string& stepName)
{
    stringstream ss;
    ss << "Step " << step << ": " << stepName;
    getLog().postMessage(new Message('I', false, ss.str(), "GpuMeshCharacter"));
}

void AbstractMesher::triangulateDomain()
{
}

void AbstractMesher::smoothMesh()
{
}
