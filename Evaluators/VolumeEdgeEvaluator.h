#ifndef GPUMESH_VOLUMEEDGEEVALUATOR
#define GPUMESH_VOLUMEEDGEEVALUATOR

#include "AbstractEvaluator.h"


class VolumeEdgeEvaluator : public AbstractEvaluator
{
public:
    VolumeEdgeEvaluator();
    virtual ~VolumeEdgeEvaluator();

    virtual double tetQuality(const glm::dvec3 vp[]) const override;

    virtual double priQuality(const glm::dvec3 vp[]) const override;

    virtual double hexQuality(const glm::dvec3 vp[]) const override;
};

#endif // GPUMESH_VOLUMEEDGEEVALUATOR
