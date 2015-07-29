#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include <functional>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/OptionMap.h"

class AbstractEvaluator;


class AbstractSmoother
{
public:
    AbstractSmoother(const std::string& smoothShader);
    virtual ~AbstractSmoother();

    virtual OptionMapDetails availableImplementations() const;

    virtual void smoothMesh(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const std::string& implementationName,
            int minIteration,
            double moveFactor,
            double gainThreshold);

    virtual void smoothMeshCpp(
            Mesh& mesh,
            AbstractEvaluator& evaluator) = 0;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            AbstractEvaluator& evaluator);


    // Mesh is garanteed to be reset to initial state after benchmarks
    virtual void benchmark(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            int minIteration,
            double moveFactor,
            double gainThreshold);


protected:
    virtual void initializeProgram(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateCpuMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateGpuMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator, bool gpu);


    int _minIteration;
    double _moveFactor;
    double _gainThreshold;

    int _smoothPassId;
    double _lastQualityMean;
    double _lastMinQuality;


    bool _initialized;
    std::string _smoothShader;
    std::string _shapeMeasureShader;
    cellar::GlProgram _smoothingProgram;

    typedef std::function<void(Mesh&, AbstractEvaluator&)> ImplementationFunc;
    OptionMap<ImplementationFunc> _implementationFuncs;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
