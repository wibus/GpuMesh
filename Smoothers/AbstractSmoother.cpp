#include "AbstractSmoother.h"

#include <chrono>
#include <iomanip>

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
    _implementationFuncs.setDefault("Serial");
    _implementationFuncs.setContent({
        {string("Serial"),  ImplementationFunc(bind(&AbstractSmoother::smoothMeshSerial, this, _1, _2))},
        {string("GLSL"),    ImplementationFunc(bind(&AbstractSmoother::smoothMeshGlsl, this, _1, _2))},
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
    mesh.updateCpuVertices();
}

void AbstractSmoother::initializeProgram(Mesh& mesh, AbstractEvaluator& evaluator)
{
    if(_initialized &&
       _modelBoundsShader == mesh.modelBoundsShaderName() &&
       _shapeMeasureShader == evaluator.shapeMeasureShader())
        return;


    getLog().postMessage(new Message('I', false,
        "Initializing smoothing compute shader", "AbstractSmoother"));

    _modelBoundsShader = mesh.modelBoundsShaderName();
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
    _smoothingProgram.addShader(GL_COMPUTE_SHADER,
        _modelBoundsShader);
    _smoothingProgram.link();

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
            evaluator.evaluateMeshQualitySerial(
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

struct SmoothBenchmarkStats
{
    string impl;
    double minQuality;
    double qualityMean;
    double qualityMeanGain;
    chrono::high_resolution_clock::rep time;
};

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

    std::vector<SmoothBenchmarkStats> statsVec;
    for(auto& impl : _implementationFuncs.details().options)
    {
        getLog().postMessage(new Message('I', false,
           "Benchmarking "+ impl +" implementation",
           "AbstractSmoother"));

        ImplementationFunc implementationFunc;
        if(_implementationFuncs.select(impl, implementationFunc))
        {
            tStart = high_resolution_clock::now();
            implementationFunc(mesh, evaluator);
            tEnd = high_resolution_clock::now();


            SmoothBenchmarkStats stats;
            stats.impl = impl;
            stats.time = (tEnd - tStart).count();
            evaluator.evaluateMeshQualitySerial(
                mesh, stats.minQuality, stats.qualityMean);
            stats.qualityMeanGain = (stats.qualityMean - initialQualityMean) /
                                        initialQualityMean;

            statsVec.push_back(stats);

            // Restore mesh vertices' initial position
            mesh.vert = verticesBackup;
            mesh.updateGpuVertices();
        }
    }

    // Get minimums for ratio computations
    double minTime = statsVec[0].time;
    double minGain = statsVec[0].qualityMeanGain;
    for(size_t i = 1; i < statsVec.size(); ++i)
    {
        minTime = glm::min(minTime, double(statsVec[i].time));
        minGain = glm::min(minGain, statsVec[i].qualityMeanGain);
    }

    // Build ratio strings
    stringstream nameStream;
    stringstream timeStream;
    stringstream normTimeStream;
    stringstream gainStream;
    stringstream normGainStream;
    for(size_t i = 0; i < statsVec.size(); ++i)
    {
        nameStream << statsVec[i].impl << ":";
        timeStream << fixed << setprecision(2) << statsVec[i].time / 1000000.0 << ":";
        normTimeStream << fixed << setprecision(2)  << statsVec[i].time / minTime << ":";
        gainStream << fixed << setprecision(4)  << statsVec[i].qualityMeanGain << ":";
        normGainStream << fixed << setprecision(4)  << statsVec[i].qualityMeanGain / minGain  << ":";
    }


    // Time ratio //
    getLog().postMessage(new Message('I', false,
       "Smoothing time ratio (ms) :\t "
        + nameStream.str() + "\t = "
        + timeStream.str() + "\t = "
        + normTimeStream.str(),
       "AbstractSmoother"));


    // Quality ratio //
    getLog().postMessage(new Message('I', false,
       "Smoothing quality gain ratio :\t "
        + nameStream.str() + "\t = "
        + gainStream.str() + "\t = "
        + normGainStream.str(),
       "AbstractSmoother"));
}
