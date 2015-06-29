#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include "DataStructures/Mesh.h"


class AbstractSmoother
{
public:
    AbstractSmoother(
            double moveFactor,
            double gainThreshold);
    virtual ~AbstractSmoother();

    virtual void smoothMesh(Mesh& mesh, AbstractEvaluator& evaluator) = 0;


protected:
    void evaluateInitialMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator);
    bool evaluateIterationMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator);

    double _moveFactor;
    double _gainThreshold;

    int _smoothPassId;
    double _lastQualityMean;
    double _lastQualityVar;
    double _lastMinQuality;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
