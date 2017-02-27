#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include <functional>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/Schedule.h"
#include "DataStructures/OptionMap.h"
#include "DataStructures/OptimizationPlot.h"

class MeshCrew;
class Schedule;


class AbstractSmoother
{
protected:
    AbstractSmoother();

public:
    virtual ~AbstractSmoother();

    virtual OptionMapDetails availableImplementations() const;

    virtual void smoothMesh(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::string& implementationName,
            const Schedule& schedule,
            OptimizationImpl& optImpl);

    virtual void smoothMeshSerial(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void smoothMeshThread(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew) = 0;


protected:
    virtual void initializeProgram(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationImpl& plotImpl) const = 0;

    bool evaluateMeshQualitySerial(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQualityThread(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQualityGlsl(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQualityCuda(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQuality(Mesh& mesh, const MeshCrew& crew, int impl);

    std::string smoothingUtilsShader() const;


    Schedule _schedule;

    int _relocPassId;
    int _globalPassId;
    double _lastPassMinQuality;
    double _lastPassQualityMean;
    double _lastIterationMinQuality;
    double _lastIterationQualityMean;

    static const int INITIAL_PASS_ID;
    static const int COMPARE_PASS_ID;

private:
    std::string _smoothingUtilsShader;

    std::vector<OptimizationPass> _optimizationPasses;
    std::chrono::high_resolution_clock::time_point _implBeginTimeStamp;

    typedef std::function<void(Mesh&, const MeshCrew&)> ImplementationFunc;
    OptionMap<ImplementationFunc> _implementationFuncs;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
