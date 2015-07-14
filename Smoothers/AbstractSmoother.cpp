#include "AbstractSmoother.h"

#include <chrono>
#include <iostream>

#include <CellarWorkbench/Misc/Log.h>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;


AbstractSmoother::AbstractSmoother(const string& smoothShader) :
    _initialized(false),
    _smoothShader(smoothShader)
{
    using namespace std::placeholders;
    _implementationFuncs = decltype(_implementationFuncs) {
        {string("C++"),  ImplementationFunc(bind(&AbstractSmoother::smoothCpuMesh, this, _1, _2))},
        {string("GLSL"), ImplementationFunc(bind(&AbstractSmoother::smoothGpuMesh, this, _1, _2))},
    };
}

AbstractSmoother::~AbstractSmoother()
{

}

std::vector<std::string> AbstractSmoother::availableImplementations() const
{
    std::vector<std::string> names;
    for(const auto& keyValue : _implementationFuncs)
        names.push_back(keyValue.first);
    return names;
}

void AbstractSmoother::smoothMesh(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        const std::string& implementationName,
        int minIteration,
        double moveFactor,
        double gainThreshold)
{
    auto it = _implementationFuncs.find(implementationName);
    if(it != _implementationFuncs.end())
    {
        _minIteration = minIteration;
        _moveFactor = moveFactor;
        _gainThreshold = gainThreshold;
        it->second(mesh, evaluator);
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Failed to find '" + implementationName + "' implementation", "AbstractSmoother"));
    }
}

void AbstractSmoother::smoothGpuMesh(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    if(!_initialized)
    {
        initializeProgram(mesh);

        _initialized = true;
    }
    else
    {
        // Absurdly make subsequent passes much more faster...
        // I guess it's because the driver put buffer back on GPU.
        // It looks like glGetBufferSubData take it out of the GPU.
        mesh.updateGpuVertices();
    }


    int vertCount = mesh.vertCount();

    _smoothingProgram.pushProgram();
    _smoothingProgram.setFloat("MoveCoeff", _moveFactor);

    auto tStart = chrono::high_resolution_clock::now();
    auto tMiddle = tStart;
    auto tEnd = tStart;

    _smoothPassId = 0;
    while(evaluateGpuMeshQuality(mesh, evaluator))
    {
        mesh.bindShaderStorageBuffers();
        glDispatchCompute(ceil(vertCount / 256.0), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    _smoothingProgram.popProgram();


    // Fetch new vertices' position
    tMiddle = chrono::high_resolution_clock::now();
    mesh.updateCpuVertices();
    tEnd = chrono::high_resolution_clock::now();


    // Display time profiling
    chrono::microseconds dtMid;
    dtMid = chrono::duration_cast<chrono::microseconds>(tMiddle - tStart);
    cout << "Total shader time = " << dtMid.count() / 1000.0 << "ms" << endl;
    chrono::microseconds dtEnd;
    dtEnd = chrono::duration_cast<chrono::microseconds>(tEnd - tMiddle);
    cout << "Get buffer time = " << dtEnd.count() / 1000.0 << "ms" << endl;
}

void AbstractSmoother::initializeProgram(Mesh& mesh)
{
    getLog().postMessage(new Message('I', false,
        "Initializing smoothing compute shader", "AbstractSmoother"));

    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _smoothShader.c_str()});
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Boundary/ElbowPipe.glsl"});
    _smoothingProgram.link();
    mesh.uploadGeometry(_smoothingProgram);
}

bool AbstractSmoother::evaluateCpuMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator)
{
    return evaluateMeshQuality(mesh, evaluator, false);
}

bool AbstractSmoother::evaluateGpuMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator)
{
    return evaluateMeshQuality(mesh, evaluator, true);
}

bool AbstractSmoother::evaluateMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator, bool gpu)
{
    bool continueSmoothing = true;
    if(_smoothPassId >= _minIteration)
    {
        double qualMean, qualMin;
        if(gpu)
        {
            evaluator.evaluateGpuMeshQuality(
                mesh, qualMin, qualMean);
        }
        else
        {
            evaluator.evaluateCpuMeshQuality(
                mesh, qualMin, qualMean);
        }

        cout << "Smooth pass number " << _smoothPassId << endl;
        cout << "Mesh minimum quality: " << qualMin << endl;
        cout << "Mesh quality mean: " << qualMean << endl;


        if(_smoothPassId > _minIteration)
        {
            continueSmoothing = (qualMean - _lastQualityMean) > _gainThreshold;
        }

        _lastQualityMean = qualMean;
        _lastMinQuality = qualMin;
    }

    ++_smoothPassId;
    return continueSmoothing;
}
