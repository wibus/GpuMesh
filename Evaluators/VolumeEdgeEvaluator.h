#ifndef GPUMESH_VOLUMEEDGEEVALUATOR
#define GPUMESH_VOLUMEEDGEEVALUATOR

#include "AbstractEvaluator.h"


class VolumeEdgeEvaluator : public AbstractEvaluator
{
public:
    VolumeEdgeEvaluator();
    virtual ~VolumeEdgeEvaluator();

    virtual double tetrahedronQuality(const glm::dvec3 verts[]) const override;

    virtual double prismQuality(const glm::dvec3 verts[]) const override;

    virtual double hexahedronQuality(const glm::dvec3 verts[]) const override;
};

#endif // GPUMESH_VOLUMEEDGEEVALUATOR
