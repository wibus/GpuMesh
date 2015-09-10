#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include <functional>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/OptionMap.h"
#include "DataStructures/OptimizationPlot.h"

class AbstractEvaluator;


struct ExclusiveDispatch
{
    ExclusiveDispatch(uint base, uint size, uint wgc) :
        base(base), size(size), workgroupCount(wgc) {}

    // Patch exclusive group range
    uint base;
    uint size;

    // Compute Dispatch Size
    uint workgroupCount;
};


class AbstractSmoother
{
public:
    AbstractSmoother();
    virtual ~AbstractSmoother();

    virtual OptionMapDetails availableImplementations() const;

    virtual void smoothMesh(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const std::string& implementationName,
            int minIteration,
            double moveFactor,
            double gainThreshold);

    virtual void smoothMeshSerial(
            Mesh& mesh,
            AbstractEvaluator& evaluator) = 0;

    virtual void smoothMeshThread(
            Mesh& mesh,
            AbstractEvaluator& evaluator) = 0;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            AbstractEvaluator& evaluator) = 0;


    // Mesh is garanteed to be reset to initial state after benchmarks
    virtual OptimizationPlot benchmark(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const std::map<std::string, bool>& activeImpls,
            int minIteration,
            double moveFactor,
            double gainThreshold,
            OptimizationPlot& outPlot);


protected:
    virtual void printImplParameters(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            OptimizationImpl& implementation) const = 0;

    virtual void initializeProgram(
            Mesh& mesh,
            AbstractEvaluator& evaluator) = 0;

    virtual void organizeDispatches(
            const Mesh& mesh,
            size_t workgroupSize,
            std::vector<ExclusiveDispatch>& dispatches) const;

    bool evaluateMeshQualitySerial(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateMeshQualityThread(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateMeshQualityGlsl(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator, int impl);


    int _minIteration;
    double _moveFactor;
    double _gainThreshold;

    int _smoothPassId;
    double _lastQualityMean;
    double _lastMinQuality;

private:
    std::chrono::high_resolution_clock::time_point _implBeginTimeStamp;
    OptimizationImpl _currentImplementation;

    typedef std::function<void(Mesh&, AbstractEvaluator&)> ImplementationFunc;
    OptionMap<ImplementationFunc> _implementationFuncs;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
