#include "AbstractSmoother.h"

#include <chrono>
#include <iomanip>

#include <CellarWorkbench/Misc/Log.h>

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "DataStructures/QualityHistogram.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Topologists/AbstractTopologist.h"

using namespace std;
using namespace cellar;


const int AbstractSmoother::INITIAL_PASS_ID = -1;
const int AbstractSmoother::COMPARE_PASS_ID = -2;

AbstractSmoother::AbstractSmoother(const installCudaFct installCuda) :
    _installCudaSmoother(installCuda),
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
        const Schedule& schedule)
{
    ImplementationFunc implementationFunc;
    if(_implementationFuncs.select(implementationName, implementationFunc))
    {
        _schedule = schedule;

        auto tStart = chrono::high_resolution_clock::now();
        implementationFunc(mesh, crew);
        auto tEnd = chrono::high_resolution_clock::now();

        auto dt = chrono::duration_cast<chrono::milliseconds>(tEnd - tStart);
        getLog().postMessage(new Message('I', true,
            "Smoothing time: " + to_string(dt.count() / 1000.0) + "s", "AbstractSmoother"));
    }
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
    QualityHistogram histogram;
    switch(impl)
    {
    case 0 :
        crew.evaluator().evaluateMeshQualitySerial(
            mesh, crew.sampler(), crew.measurer(), histogram);
        break;
    case 1 :
        crew.evaluator().evaluateMeshQualityThread(
            mesh, crew.sampler(), crew.measurer(), histogram);
        break;
    case 2 :
        crew.evaluator().evaluateMeshQualityGlsl(
            mesh, crew.sampler(), crew.measurer(), histogram);
        break;
    case 3 :
        crew.evaluator().evaluateMeshQualityCuda(
            mesh, crew.sampler(), crew.measurer(), histogram);
        break;
    }


    bool continueSmoothing = true;
    auto statsNow = chrono::high_resolution_clock::now();

    if(_relocPassId == INITIAL_PASS_ID)
    {
        getLog().postMessage(new Message('I', true,
            std::string("Initial mesh quality : ") +
            "min=" + to_string(histogram.minimumQuality()) +
            "\t mean=" + to_string(histogram.geometricMean()),
            "AbstractSmoother"));

        _lastPassMinQuality = histogram.minimumQuality();
        _lastPassGeomQuality = histogram.geometricMean();
        _lastIterationMinQuality = histogram.minimumQuality();
        _lastIterationGeomQuality = histogram.geometricMean();

        _currentImplementation.passes.clear();
        _implBeginTimeStamp = statsNow;

        _relocPassId = 0;
        _globalPassId = 1;
    }
    else if(_relocPassId == COMPARE_PASS_ID)
    {
        getLog().postMessage(new Message('I', true,
            std::string("Topo/Reloc pass quality " +
                  to_string(_globalPassId) + " : ") +
            "min=" + to_string(histogram.minimumQuality()) +
            "\t mean=" + to_string(histogram.geometricMean()),
            "AbstractSmoother"));

        double minGain = histogram.minimumQuality() - _lastPassMinQuality;
        double geomGain = histogram.geometricMean() - _lastPassGeomQuality;

        if(_schedule.autoPilotEnabled)
        {
            continueSmoothing = (geomGain > _schedule.minQualThreshold) ||
                                (minGain  > _schedule.qualMeanThreshold);
        }
        else
        {
            continueSmoothing = _globalPassId < _schedule.globalPassCount;
        }

        _lastPassMinQuality = histogram.minimumQuality();
        _lastPassGeomQuality = histogram.geometricMean();

        ++_globalPassId;
        _relocPassId = 0;
    }
    else
    {
        getLog().postMessage(new Message('I', true,
            "Smooth pass " + to_string(_globalPassId) + "|" +
                             to_string(_relocPassId) + " : " +
            "min=" + to_string(histogram.minimumQuality()) +
            "\t mean=" + to_string(histogram.geometricMean()),
            "AbstractSmoother"));

        double minGain = histogram.minimumQuality() - _lastIterationMinQuality;
        double geomGain = histogram.geometricMean() - _lastIterationGeomQuality;

        if(_schedule.autoPilotEnabled && _relocPassId > 0)
        {
            continueSmoothing = (geomGain > _schedule.minQualThreshold) ||
                                (minGain  > _schedule.qualMeanThreshold);
        }
        else
        {
            continueSmoothing = _relocPassId < _schedule.nodeRelocationsPassCount;
        }

        OptimizationPass stats;
        stats.histogram = histogram;
        stats.timeStamp = (statsNow - _implBeginTimeStamp).count() / 1.0e9;
        _currentImplementation.passes.push_back(stats);

        _lastIterationMinQuality = histogram.minimumQuality();
        _lastIterationGeomQuality = histogram.geometricMean();

        ++_relocPassId;
    }

    return continueSmoothing;
}

struct SmoothBenchmarkStats
{
    string impl;
    double totalSeconds;
    QualityHistogram histogram;
    double qualityGain;
};

void AbstractSmoother::benchmark(
        Mesh& mesh,
        const MeshCrew& crew,
        const map<string, bool>& activeImpls,
        const Schedule& schedule,
        OptimizationPlot& outPlot)
{
    _schedule = schedule;
    initializeProgram(mesh, crew);

    printOptimisationParameters(mesh, outPlot);
    // TODO print evaluator parameters
    // TODO print sampler parameters

    QualityHistogram initialHistogram;
    crew.evaluator().evaluateMeshQualityThread(
        mesh, crew.sampler(), crew.measurer(),
        initialHistogram);

    outPlot.setInitialHistogram(initialHistogram);

    // We must make a copy of the vertices in order to
    // restore mesh's vertices after each implementation.
    Mesh meshBackup = mesh;

    // Make a copy of the smoothed vertices from the firts valid
    // implementation to show the results of the smoothing process.
    Mesh smoothedMesh;

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
            std::string topo = " (topo=";
            if(_schedule.topoOperationEnabled)
                topo += "on)";
            else
                topo += "off)";

            getLog().postMessage(new Message('I', false,
               "Benchmarking "+ impl + topo +" implementation",
               "AbstractSmoother"));


            _currentImplementation = OptimizationImpl();
            _currentImplementation.name = impl + topo;

            implementationFunc(mesh, crew);

            SmoothBenchmarkStats stats;
            crew.evaluator().evaluateMeshQualityThread(
                mesh, crew.sampler(), crew.measurer(),
                stats.histogram);

            outPlot.addImplementation(_currentImplementation);

            stats.impl = impl;
            stats.totalSeconds = _currentImplementation.passes.back().timeStamp -
                                 _currentImplementation.passes.front().timeStamp;
            stats.qualityGain = stats.histogram.computeGain(initialHistogram);
            statsVec.push_back(stats);

            if(smoothedMesh.verts.empty())
                smoothedMesh = mesh;

            // Restore mesh vertices' initial position
            mesh = meshBackup;
        }
        else
        {
            getLog().postMessage(new Message('E', false,
               "Requested implementation not found: " + impl,
               "AbstractSmoother"));
        }
    }

    if(!smoothedMesh.verts.empty())
    {
        mesh = smoothedMesh;
    }

    // Get minimums for ratio computations
    double minTime = statsVec[0].totalSeconds;
    double minGain = statsVec[0].qualityGain;
    for(size_t i = 1; i < statsVec.size(); ++i)
    {
        minTime = glm::min(minTime, double(statsVec[i].totalSeconds));
        minGain = glm::min(minGain, statsVec[i].qualityGain);
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
        gainStream << fixed << setprecision(6)  << statsVec[i].qualityGain << ":";
        normGainStream << fixed << setprecision(6)  << statsVec[i].qualityGain / minGain  << ":";
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
