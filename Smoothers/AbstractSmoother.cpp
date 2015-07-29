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
        {string("C++"),  ImplementationFunc(bind(&AbstractSmoother::smoothMeshCpp, this, _1, _2))},
        {string("GLSL"), ImplementationFunc(bind(&AbstractSmoother::smoothMeshGlsl, this, _1, _2))},
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

void AbstractSmoother::smoothMeshGlsl(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    initializeProgram(mesh, evaluator);

    // There's no need to upload vertices again, but absurdly
    // this makes subsequent passes much more faster...
    // I guess it's because the driver put buffer back on GPU.
    // It looks like glGetBufferSubData take it out of the GPU.
    mesh.updateGpuVertices();


    using chrono::high_resolution_clock;
    high_resolution_clock::time_point tStart, tMiddle, tEnd;
    tStart = high_resolution_clock::now();

    _smoothPassId = 0;
    _smoothingProgram.pushProgram();
    _smoothingProgram.setFloat("MoveCoeff", _moveFactor);
    mesh.bindShaderStorageBuffers();
    int vertCount = mesh.vertCount();
    while(evaluateGpuMeshQuality(mesh, evaluator))
    {
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
    if(_initialized && _shapeMeasureShader == evaluator.shapeMeasureShader())
        return;


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


    _initialized = true;
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
            evaluator.evaluateMeshQualityGlsl(
                mesh, qualMin, qualMean);
        }
        else
        {
            evaluator.evaluateMeshQualityCpp(
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

void AbstractSmoother::benchmark(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        int minIteration,
        double moveFactor,
        double gainThreshold)
{
    _minIteration = minIteration;
    _moveFactor = moveFactor;
    _gainThreshold = gainThreshold;
    initializeProgram(mesh, evaluator);

    double initialMinQuality = 0.0;
    double initialQualityMean = 0.0;
    evaluator.evaluateMeshQualityGlsl(
        mesh, initialMinQuality, initialQualityMean);

    // We must make a copy of the vertices in order to
    // restore mesh's vertices after benchmarks.
    auto verticesBackup = mesh.vert;

    using std::chrono::high_resolution_clock;
    high_resolution_clock::time_point tStart;
    high_resolution_clock::time_point tEnd;


    // C++ IMPLEMENTATION //
    getLog().postMessage(new Message('I', false,
       "Benchmarking C++ implementation",
       "AbstractSmoother"));

    tStart = high_resolution_clock::now();
    smoothMeshCpp(mesh, evaluator);
    tEnd = high_resolution_clock::now();
    high_resolution_clock::duration cppTime = (tEnd - tStart);

    double cppMinQuality = 0.0;
    double cppQualityMean = 0.0;
    evaluator.evaluateMeshQualityCpp(
        mesh, cppMinQuality, cppQualityMean);

    // Restore mesh vertices' initial position
    mesh.vert = verticesBackup;
    mesh.updateGpuVertices();


    // GLSL IMPLEMENTATION //
    getLog().postMessage(new Message('I', false,
       "Benchmarking GLSL implementation",
       "AbstractSmoother"));

    tStart = high_resolution_clock::now();
    smoothMeshGlsl(mesh, evaluator);
    tEnd = high_resolution_clock::now();
    high_resolution_clock::duration glslTime = (tEnd - tStart);

    double glslMinQuality = 0.0;
    double glslQualityMean = 0.0;
    evaluator.evaluateMeshQualityCpp(
        mesh, glslMinQuality, glslQualityMean);

    // Restore mesh vertices' initial position
    mesh.vert = verticesBackup;
    mesh.updateGpuVertices();



    // Time ratio //
    auto cppNano = cppTime.count();
    auto glslNano = glslTime.count();
    double minNano = glm::min(cppNano, glslNano);

    double cppRatio = cppNano / minNano;
    double glslRatio = glslNano / minNano;

    getLog().postMessage(new Message('I', false,
       "Smoothing time ratio (ms) : \tC++:GLSL \t= " +
        to_string(cppNano / 1000000.0) + ":" + to_string(glslNano / 1000000.0) + " \t= " +
        to_string(cppRatio) + ":" + to_string(glslRatio),
       "AbstractSmoother"));



    // Quality ratio //
    auto cppQualityGain = (cppQualityMean - initialQualityMean) / initialQualityMean;
    auto glslQualityGain = (glslQualityMean - initialQualityMean) / initialQualityMean;
    double minGain = glm::min(cppQualityGain, glslQualityGain);

    double cppGainRatio = cppQualityGain / minGain;
    double glslGainRatio = glslQualityGain / minGain;

    getLog().postMessage(new Message('I', false,
       "Smoothing quality gain ratio : \tC++:GLSL \t= " +
        to_string(cppQualityGain) + ":" + to_string(glslQualityGain) + "\t = " +
        to_string(cppGainRatio) + ":" + to_string(glslGainRatio),
       "AbstractSmoother"));
}
