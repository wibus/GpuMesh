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
    Mesh& _mesh;
    double _moveFactor;
    double _gainThreshold;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
