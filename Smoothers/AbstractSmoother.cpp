#include "AbstractSmoother.h"

#include <chrono>
#include <iomanip>

#include <CellarWorkbench/Misc/Log.h>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;

AbstractSmoother::AbstractSmoother() :
    _implementationFuncs("Smoothing Implementations")
{
    using namespace std::placeholders;
    _implementationFuncs.setDefault("Thread");
    _implementationFuncs.setContent({
        {string("Serial"),  ImplementationFunc(bind(&AbstractSmoother::smoothMeshSerial, this, _1, _2))},
        {string("Thread"),  ImplementationFunc(bind(&AbstractSmoother::smoothMeshThread, this, _1, _2))},
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

bool AbstractSmoother::evaluateMeshQualitySerial(Mesh& mesh, AbstractEvaluator& evaluator)
{
    return evaluateMeshQuality(mesh, evaluator, 0);
}

bool AbstractSmoother::evaluateMeshQualityThread(Mesh& mesh, AbstractEvaluator& evaluator)
{
    return evaluateMeshQuality(mesh, evaluator, 1);
}

bool AbstractSmoother::evaluateMeshQualityGlsl(Mesh& mesh, AbstractEvaluator& evaluator)
{
    return evaluateMeshQuality(mesh, evaluator, 2);
}

bool AbstractSmoother::evaluateMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator, int impl)
{
    double qualMean, qualMin;
    switch(impl)
    {
    case 0 :
        evaluator.evaluateMeshQualitySerial(
            mesh, qualMin, qualMean);
        break;
    case 1 :
        evaluator.evaluateMeshQualityThread(
            mesh, qualMin, qualMean);
        break;
    case 2 :
        evaluator.evaluateMeshQualityGlsl(
            mesh, qualMin, qualMean);
        break;
    }

    getLog().postMessage(new Message('I', true,
        "Smooth pass " + to_string(_smoothPassId) + ": " +
        "min=" + to_string(qualMin) + "\t mean=" + to_string(qualMean), "AbstractSmoother"));


    bool continueSmoothing = true;
    if(_smoothPassId > _minIteration)
    {
        continueSmoothing = (qualMean - _lastQualityMean) > _gainThreshold;
    }

    auto statsNow = chrono::high_resolution_clock::now();
    if(_smoothPassId == 0)
    {
        _implBeginTimeStamp = statsNow;
        _currentPassVect.clear();
    }
    OptimizationPass stats;
    stats.timeStamp = (statsNow - _implBeginTimeStamp).count() / 1.0e9;
    stats.minQuality = qualMin;
    stats.qualityMean = qualMean;
    _currentPassVect.push_back(stats);


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
    chrono::high_resolution_clock::rep time;
};

OptimizationPlot AbstractSmoother::benchmark(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        const map<string, bool>& activeImpls,
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
    evaluator.evaluateMeshQualityThread(
        mesh, initialMinQuality, initialQualityMean);

    // We must make a copy of the vertices in order to
    // restore mesh's vertices after benchmarks.
    auto verticesBackup = mesh.verts;

    using std::chrono::high_resolution_clock;
    high_resolution_clock::time_point tStart;
    high_resolution_clock::time_point tEnd;

    OptimizationPlot plotModel("Smoothing implementation benchmarks");

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

        getLog().postMessage(new Message('I', false,
           "Benchmarking "+ impl +" implementation",
           "AbstractSmoother"));

        ImplementationFunc implementationFunc;
        if(_implementationFuncs.select(impl, implementationFunc))
        {
            tStart = high_resolution_clock::now();
            implementationFunc(mesh, evaluator);
            tEnd = high_resolution_clock::now();

            plotModel.addCurve(impl, _currentPassVect);


            SmoothBenchmarkStats stats;
            stats.impl = impl;
            stats.time = (tEnd - tStart).count();
            evaluator.evaluateMeshQualityThread(
                mesh, stats.minQuality, stats.qualityMean);
            stats.qualityMeanGain = (stats.qualityMean - initialQualityMean) /
                                        initialQualityMean;

            statsVec.push_back(stats);

            // Restore mesh vertices' initial position
            mesh.verts = verticesBackup;
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

    return plotModel;
}
