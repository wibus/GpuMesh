#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include "DataStructures/Mesh.h"


class AbstractSmoother
{
public:
    AbstractSmoother(Mesh& mesh, double moveFactor, double gainThreshold);
    virtual ~AbstractSmoother();

    virtual void smoothMesh() = 0;


protected:
    void evaluateInitialMeshQuality();
    bool evaluateIterationMeshQuality();

    Mesh& _mesh;
    double _moveFactor;
    double _gainThreshold;

    int _smoothPassId;
    double _lastQualityMean;
    double _lastQualityVar;
    double _lastMinQuality;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
