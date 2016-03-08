#include "AbstractSmoother.h"

#include <chrono>
#include <iomanip>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;


AbstractSmoother::AbstractSmoother(const installCudaFct installCuda) :
    _installCuda(installCuda),
    _smoothingUtilsShader(":/glsl/compute/Smoothing/Utils.glsl"),
    _implementationFuncs("Smoothing Implementations")
{
    using namespace std::placeholders;
    _implementationFuncs.setDefault("Thread");
    _implementationFuncs.setContent({
        {string("Serial"),  ImplementationFunc(bind(&AbstractSmoother::smoothMeshSerial, this, _1, _2))},
        {string("Thread"),  ImplementationFunc(bind(&AbstractSmoother::smoothMeshThread, this, _1, _2))},
        {string("GLSL"),    ImplementationFunc(bind(&AbstractSmoother::smoothMeshGlsl,   this, _1, _2))},
        {string("CUDA"),    ImplementationFunc(bind(&AbstractSmoother::smoothMeshCuda,   this, _1, _2))},
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
        const MeshCrew& crew,
        const std::string& implementationName,
        int minIteration,
        double moveFactor,
        double gainThreshold)
{
    ImplementationFunc implementationFunc;
    if(_implementationFuncs.select(implementationName, implementationFunc))
    {
        _minIteration = minIteration;
        _moveCoeff = moveFactor;
        _gainThreshold = gainThreshold;

        auto tStart = chrono::high_resolution_clock::now();
        implementationFunc(mesh, crew);
        auto tEnd = chrono::high_resolution_clock::now();

        auto dt = chrono::duration_cast<chrono::milliseconds>(tEnd - tStart);
        getLog().postMessage(new Message('I', true,
            "Smoothing time: " + to_string(dt.count() / 1000.0) + "s", "AbstractSmoother"));
    }
}

void AbstractSmoother::organizeDispatches(
        const Mesh& mesh,
        size_t workgroupSize,
        std::vector<IndependentDispatch>& dispatches) const
{
    size_t groupCount = mesh.independentGroups.size();

    size_t base = 0;
    for(size_t i=0; i < groupCount; ++i)
    {
        size_t count = mesh.independentGroups[i].size();
        size_t wg = glm::ceil(count / double(workgroupSize));
        dispatches.push_back(IndependentDispatch(base, count, wg));

        base += count;
    }
}

bool AbstractSmoother::isSmoothable(
        const Mesh& mesh,
        size_t vId) const
{
    const MeshTopo& topo = mesh.topos[vId];
    if(topo.isFixed)
        return false;

    size_t neigElemCount = topo.neighborElems.size();
    if(neigElemCount == 0)
        return false;

    return true;
}

bool AbstractSmoother::evaluateMeshQualitySerial(Mesh& mesh,  const MeshCrew& crew)
{
    return evaluateMeshQuality(mesh, crew, 0);
}

bool AbstractSmoother::evaluateMeshQualityThread(Mesh& mesh,  const MeshCrew& crew)
{
    return evaluateMeshQuality(mesh, crew, 1);
}

bool AbstractSmoother::evaluateMeshQualityGlsl(Mesh& mesh,  const MeshCrew& crew)
{
    return evaluateMeshQuality(mesh, crew, 2);
}

bool AbstractSmoother::evaluateMeshQualityCuda(Mesh& mesh,  const MeshCrew& crew)
{
    return evaluateMeshQuality(mesh, crew, 3);
}

bool AbstractSmoother::evaluateMeshQuality(Mesh& mesh,  const MeshCrew& crew, int impl)
{
    double qualMean, qualMin;
    switch(impl)
    {
    case 0 :
        crew.evaluator().evaluateMeshQualitySerial(
            mesh, crew.sampler(), crew.measurer(), qualMin, qualMean);
        break;
    case 1 :
        crew.evaluator().evaluateMeshQualityThread(
            mesh, crew.sampler(), crew.measurer(), qualMin, qualMean);
        break;
    case 2 :
        crew.evaluator().evaluateMeshQualityGlsl(
            mesh, crew.sampler(), crew.measurer(), qualMin, qualMean);
        break;
    case 3 :
        crew.evaluator().evaluateMeshQualityCuda(
            mesh, crew.sampler(), crew.measurer(), qualMin, qualMean);
        break;
    }

    getLog().postMessage(new Message('I', true,
        "Smooth pass " + to_string(_smoothPassId) + ": " +
        "min=" + to_string(qualMin) + "\t mean=" + to_string(qualMean), "AbstractSmoother"));


    bool continueSmoothing = true;
    if(_smoothPassId > _minIteration)
    {
        double minGain = qualMin  - _lastMinQuality;
        double meanGain = qualMean - _lastQualityMean;
        double summedGain = minGain + meanGain;

        continueSmoothing = minGain > _gainThreshold ||
                            meanGain > _gainThreshold ||
                            summedGain > _gainThreshold;
    }

    auto statsNow = chrono::high_resolution_clock::now();
    if(_smoothPassId == 0)
    {
        _implBeginTimeStamp = statsNow;
        _currentImplementation.passes.clear();
    }
    OptimizationPass stats;
    stats.timeStamp = (statsNow - _implBeginTimeStamp).count() / 1.0e9;
    stats.minQuality = qualMin;
    stats.qualityMean = qualMean;
    _currentImplementation.passes.push_back(stats);


    _lastQualityMean = qualMean;
    _lastMinQuality = qualMin;

    ++_smoothPassId;
    return continueSmoothing;
}

struct SmoothBenchmarkStats
{
    string impl;
    double minQuality;
    double qualityMean;
    double qualityMeanGain;
    double totalSeconds;
};

void AbstractSmoother::benchmark(
        Mesh& mesh,
        const MeshCrew& crew,
        const map<string, bool>& activeImpls,
        int minIteration,
        double moveFactor,
        double gainThreshold,
        OptimizationPlot& outPlot)
{
    _minIteration = minIteration;
    _moveCoeff = moveFactor;
    _gainThreshold = gainThreshold;
    initializeProgram(mesh, crew);

    printSmoothingParameters(mesh, outPlot);
    // TODO print evaluator parameters
    // TODO print sampler parameters

    double initialMinQuality = 0.0;
    double initialQualityMean = 0.0;
    crew.evaluator().evaluateMeshQualityThread(
        mesh, crew.sampler(), crew.measurer(),
        initialMinQuality, initialQualityMean);

    // We must make a copy of the vertices in order to
    // restore mesh's vertices after each implementation.
    auto verticesBackup = mesh.verts;

    // Make a copy of the smoothed vertices from the firts valid
    // implementation to show the results of the smoothing process.
    vector<MeshVert> smoothedVertices;

    std::vector<SmoothBenchmarkStats> statsVec;
    for(auto& impl : _implementationFuncs.details().options)
    {
        auto activeIt = activeImpls.find(impl);
        if(activeIt != activeImpls.end())
        {
            bool isActive = activeIt->second;

            if(!isActive)
                continue;
        }
        else
        {
            getLog().postMessage(new Message('W', false,
               "No active state defined for " + impl +
               ". Skipping this implementation...",
               "AbstractSmoother"));
            continue;
        }

        ImplementationFunc implementationFunc;
        if(_implementationFuncs.select(impl, implementationFunc))
        {
            getLog().postMessage(new Message('I', false,
               "Benchmarking "+ impl +" implementation",
               "AbstractSmoother"));

            _currentImplementation = OptimizationImpl();
            _currentImplementation.name = impl;

            implementationFunc(mesh, crew);

            outPlot.addImplementation(_currentImplementation);


            SmoothBenchmarkStats stats;
            crew.evaluator().evaluateMeshQualityThread(
                mesh, crew.sampler(), crew.measurer(),
                stats.minQuality, stats.qualityMean);

            stats.impl = impl;
            stats.totalSeconds = _currentImplementation.passes.back().timeStamp -
                                 _currentImplementation.passes.front().timeStamp;
            stats.qualityMeanGain = (stats.qualityMean - initialQualityMean) /
                                        initialQualityMean;
            statsVec.push_back(stats);


            // Keep a copy of the smoothed vertices
            if(smoothedVertices.empty())
                smoothedVertices = mesh.verts;

            // Restore mesh vertices' initial position
            mesh.verts = verticesBackup;
            mesh.updateVerticesFromCpu();
        }
        else
        {
            getLog().postMessage(new Message('E', false,
               "Requested implementation not found: " + impl,
               "AbstractSmoother"));
        }
    }

    if(!smoothedVertices.empty())
    {
        mesh.verts = smoothedVertices;
        mesh.updateVerticesFromCpu();
    }

    // Get minimums for ratio computations
    double minTime = statsVec[0].totalSeconds;
    double minGain = statsVec[0].qualityMeanGain;
    for(size_t i = 1; i < statsVec.size(); ++i)
    {
        minTime = glm::min(minTime, double(statsVec[i].totalSeconds));
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
        timeStream << fixed << setprecision(2) << statsVec[i].totalSeconds * 1000.0 << ":";
        normTimeStream << fixed << setprecision(2)  << statsVec[i].totalSeconds / minTime << ":";
        gainStream << fixed << setprecision(6)  << statsVec[i].qualityMeanGain << ":";
        normGainStream << fixed << setprecision(6)  << statsVec[i].qualityMeanGain / minGain  << ":";
    }
    string nameString = nameStream.str(); nameString.back() = ' ';
    string timeString = timeStream.str(); timeString.back() = ' ';
    string normTimeString = normTimeStream.str(); normTimeString.back() = ' ';
    string gainString = gainStream.str(); gainString.back() = ' ';
    string normGainString = normGainStream.str(); normGainString.back() = ' ';


    // Time ratio //
    getLog().postMessage(new Message('I', false,
       "Smoothing time ratio (ms) :\t "
        + nameString + "\t = "
        + timeString + "\t = "
        + normTimeString,
       "AbstractSmoother"));


    // Quality ratio //
    getLog().postMessage(new Message('I', false,
       "Smoothing quality gain ratio :\t "
        + nameString + "\t = "
        + gainString + "\t = "
        + normGainString,
       "AbstractSmoother"));
}

std::string AbstractSmoother::smoothingUtilsShader() const
{
    return _smoothingUtilsShader;
}
