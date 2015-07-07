#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


class AbstractSmoother
{
public:
    AbstractSmoother(
            int minIteration,
            double moveFactor,
            double gainThreshold,
            const std::string& smoothShader);
    virtual ~AbstractSmoother();

    virtual void smoothCpuMesh(Mesh& mesh, AbstractEvaluator& evaluator) = 0;
    virtual void smoothGpuMesh(Mesh& mesh, AbstractEvaluator& evaluator);


protected:
    bool evaluateCpuMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateGpuMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator, bool gpu);

    virtual void initializeProgram(Mesh& mesh);

    int _minIteration;
    double _moveFactor;
    double _gainThreshold;

    int _smoothPassId;
    double _lastQualityMean;
    double _lastMinQuality;


    bool _initialized;
    std::string _smoothShader;
    cellar::GlProgram _smoothingProgram;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
