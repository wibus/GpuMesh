#include "AbstractSmoother.h"

#include <chrono>
#include <iostream>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;


AbstractSmoother::AbstractSmoother(
        int minIteration,
        double moveFactor,
        double gainThreshold,
        const string& smoothShader) :
    _minIteration(minIteration),
    _moveFactor(moveFactor),
    _gainThreshold(gainThreshold),
    _initialized(false),
    _smoothShader(smoothShader)
{
}

AbstractSmoother::~AbstractSmoother()
{

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

void AbstractSmoother::smoothGpuMesh(Mesh& mesh, AbstractEvaluator& evaluator)
{
    GLuint vertSsbo = mesh.glBuffer(EMeshBuffer::VERT);
    GLuint topoSsbo = mesh.glBuffer(EMeshBuffer::TOPO);
    GLuint neigSsbo = mesh.glBuffer(EMeshBuffer::NEIG);

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
    _smoothingProgram.setInt("VertCount", vertCount);
    _smoothingProgram.setFloat("MoveCoeff", _moveFactor);

    auto tStart = chrono::high_resolution_clock::now();
    auto tMiddle = tStart;
    auto tEnd = tStart;

    _smoothPassId = 0;
    while(evaluateGpuMeshQuality(mesh, evaluator))
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vertSsbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, topoSsbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, neigSsbo);

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
    cout << "Initializing Laplacian smoothing compute shader" << endl;
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, _smoothShader);
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        ":/shaders/compute/Boundary/ElbowPipe.glsl");
    _smoothingProgram.link();

}
