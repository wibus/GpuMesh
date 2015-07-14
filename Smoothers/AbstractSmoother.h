#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include <map>
#include <functional>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"

class AbstractEvaluator;


class AbstractSmoother
{
public:
    AbstractSmoother(const std::string& smoothShader);
    virtual ~AbstractSmoother();

    virtual std::vector<std::string> availableImplementations() const;

    virtual void smoothMesh(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const std::string& implementationName,
            int minIteration,
            double moveFactor,
            double gainThreshold);

    virtual void smoothCpuMesh(
            Mesh& mesh,
            AbstractEvaluator& evaluator) = 0;

    virtual void smoothGpuMesh(
            Mesh& mesh,
            AbstractEvaluator& evaluator);


protected:
    virtual void initializeProgram(Mesh& mesh);
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
    cellar::GlProgram _smoothingProgram;

    typedef std::function<void(Mesh&, AbstractEvaluator&)> ImplementationFunc;
    std::map<std::string, ImplementationFunc> _implementationFuncs;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
