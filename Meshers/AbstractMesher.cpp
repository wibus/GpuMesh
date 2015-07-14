#include "AbstractMesher.h"

#include <chrono>

#include <CellarWorkbench/Misc/Log.h>

using namespace std;
using namespace cellar;

AbstractMesher::AbstractMesher()
{

}

AbstractMesher::~AbstractMesher()
{

}

std::vector<std::string> AbstractMesher::availableMeshModels() const
{
    std::vector<std::string> names;
    for(const auto& keyValue : _modelFuncs)
        names.push_back(keyValue.first);
    return names;
}

void AbstractMesher::generateMesh(
        Mesh& mesh,
        const string& modelName,
        size_t vertexCount)
{

    auto it = _modelFuncs.find(modelName);
    if(it != _modelFuncs.end())
    {
        chrono::high_resolution_clock::time_point startTime, endTime;
        startTime = chrono::high_resolution_clock::now();
        it->second(mesh, vertexCount);
        endTime = chrono::high_resolution_clock::now();

        chrono::microseconds dt;
        dt = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        getLog().postMessage(new Message('I', false,
            "Total meshing time: " + to_string(dt.count() / 1000.0) + "ms",
            "AbstractMesher"));

        getLog().postMessage(new Message('I', false,
            "Elements / Vertices = " + to_string(mesh.elemCount()) +
                               " / " + to_string(mesh.vertCount()) + " = " +
            to_string(mesh.elemCount()  / (double) mesh.vertCount()),
            "AbstractMesher"));
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + modelName + "' model",
            "AbstractMesher"));
    }
}
