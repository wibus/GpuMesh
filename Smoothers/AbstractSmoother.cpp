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
    _implementationFuncs.setDefault("GLSL");
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
        const Schedule& schedule,
        OptimizationImpl& optImpl)
{
    ImplementationFunc implementationFunc;
    if(_implementationFuncs.select(implementationName, implementationFunc))
    {
        _schedule = schedule;        

        printOptimisationParameters(mesh, optImpl);

        auto tStart = chrono::high_resolution_clock::now();
        implementationFunc(mesh, crew);
        auto tEnd = chrono::high_resolution_clock::now();

        optImpl.passes = _optimizationPasses;

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

        _optimizationPasses.clear();
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
        _optimizationPasses.push_back(stats);

        _lastIterationMinQuality = histogram.minimumQuality();
        _lastIterationGeomQuality = histogram.geometricMean();

        ++_relocPassId;
    }

    return continueSmoothing;
}

std::string AbstractSmoother::smoothingUtilsShader() const
{
    return _smoothingUtilsShader;
}
