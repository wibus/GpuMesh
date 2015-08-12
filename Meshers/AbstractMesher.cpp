#include "AbstractMesher.h"

#include <chrono>

#include <CellarWorkbench/Misc/Log.h>

using namespace std;
using namespace cellar;

AbstractMesher::AbstractMesher() :
    _modelFuncs("Mesh Models")
{

}

AbstractMesher::~AbstractMesher()
{

}

OptionMapDetails AbstractMesher::availableMeshModels() const
{
    return _modelFuncs.details();
}

void AbstractMesher::generateMesh(
        Mesh& mesh,
        const string& modelName,
        size_t vertexCount)
{
    ModelFunc modelFunc;
    if(_modelFuncs.select(modelName, modelFunc))
    {
        chrono::high_resolution_clock::time_point startTime, endTime;
        startTime = chrono::high_resolution_clock::now();
        modelFunc(mesh, vertexCount);
        endTime = chrono::high_resolution_clock::now();

        chrono::microseconds dt;
        dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        getLog().postMessage(new Message('I', false,
            "Total meshing time: " + to_string(dt.count() / 1000.0) + "ms",
            "AbstractMesher"));
    }
}
