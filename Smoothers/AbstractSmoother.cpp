#include "AbstractSmoother.h"

#include <chrono>

#include <CellarWorkbench/Misc/Log.h>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;


AbstractSmoother::AbstractSmoother(const string& smoothShader) :
    _initialized(false),
    _smoothShader(smoothShader),
    _implementationFuncs("Smoothing Implementations")
{
    using namespace std::placeholders;
    _implementationFuncs.setDefault("C++");
    _implementationFuncs.setContent({
        {string("C++"),  ImplementationFunc(bind(&AbstractSmoother::smoothCpuMesh, this, _1, _2))},
        {string("GLSL"), ImplementationFunc(bind(&AbstractSmoother::smoothGpuMesh, this, _1, _2))},
    });
}

AbstractSmoother::~AbstractSmoother()
{

}

OptionMapDetails AbstractSmoother::availableImplementations() const
{
    return _implementationFuncs.details();
}

void AbstractSmoother::smoothMesh(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        const std::string& implementationName,
        int minIteration,
        double moveFactor,
        double gainThreshold)
{
    ImplementationFunc implementationFunc;
    if(_implementationFuncs.select(implementationName, implementationFunc))
    {
        _minIteration = minIteration;
        _moveFactor = moveFactor;
        _gainThreshold = gainThreshold;

        auto tStart = chrono::high_resolution_clock::now();
        implementationFunc(mesh, evaluator);
        auto tEnd = chrono::high_resolution_clock::now();

        auto dt = chrono::duration_cast<chrono::milliseconds>(tEnd - tStart);
        getLog().postMessage(new Message('I', true,
            "Smoothing time: " + to_string(dt.count() / 1000.0) + "s", "AbstractSmoother"));
    }
}

void AbstractSmoother::smoothGpuMesh(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    if(!_initialized)
    {
        initializeProgram(mesh, evaluator);

        _initialized = true;
    }
    else
    {
        if(_shapeMeasureShader != evaluator.shapeMeasureShader())
        {
            initializeProgram(mesh, evaluator);
        }

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
    dtMid = chrono::duration_cast<chrono::milliseconds>(tMiddle - tStart);
    getLog().postMessage(new Message('I', true,
        "Total shader time = " + to_string(dtMid.count() / 1000.0) + "ms", "AbstractSmoother"));
    chrono::microseconds dtEnd;
    dtEnd = chrono::duration_cast<chrono::milliseconds>(tEnd - tMiddle);
    getLog().postMessage(new Message('I', true,
        "Get buffer time = " + to_string(dtEnd.count() / 1000.0) + "ms", "AbstractSmoother"));
}

void AbstractSmoother::initializeProgram(Mesh& mesh, AbstractEvaluator& evaluator)
{
    getLog().postMessage(new Message('I', false,
        "Initializing smoothing compute shader", "AbstractSmoother"));

    _shapeMeasureShader = evaluator.shapeMeasureShader();

    _smoothingProgram.clearShaders();
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _smoothShader.c_str()});
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Quality/QualityInterface.glsl"});
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _shapeMeasureShader.c_str()});
    _smoothingProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Boundary/ElbowPipe.glsl"});
    _smoothingProgram.link();\

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

        getLog().postMessage(new Message('I', true,
            "Smooth pass " + to_string(_smoothPassId) + ": " +
            "min=" + to_string(qualMin) + "\t mean=" + to_string(qualMean), "AbstractSmoother"));


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
