#ifndef GPUMESH_INSPHEREEDGEEVALUATOR
#define GPUMESH_INSPHEREEDGEEVALUATOR

#include "AbstractEvaluator.h"


class InsphereEdgeEvaluator : public AbstractEvaluator
{
public:
    InsphereEdgeEvaluator();
    virtual ~InsphereEdgeEvaluator();

    virtual double tetrahedronQuality(const glm::dvec3 verts[]) const override;

    virtual double prismQuality(const glm::dvec3 verts[]) const override;

    virtual double hexahedronQuality(const glm::dvec3 verts[]) const override;
};

#endif // GPUMESH_INSPHEREEDGEEVALUATOR
