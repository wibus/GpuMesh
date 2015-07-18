#ifndef GPUMESH_INSPHEREEDGEEVALUATOR
#define GPUMESH_INSPHEREEDGEEVALUATOR

#include "AbstractEvaluator.h"


class InsphereEdgeEvaluator : public AbstractEvaluator
{
public:
    InsphereEdgeEvaluator();
    virtual ~InsphereEdgeEvaluator();

    virtual double tetQuality(const glm::dvec3 verts[]) const override;

    virtual double priQuality(const glm::dvec3 verts[]) const override;

    virtual double hexQuality(const glm::dvec3 verts[]) const override;
};

#endif // GPUMESH_INSPHEREEDGEEVALUATOR
