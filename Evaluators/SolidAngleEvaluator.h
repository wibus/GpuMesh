#ifndef GPUMESH_SOLIDANGLEEVALUATOR
#define GPUMESH_SOLIDANGLEEVALUATOR

#include "AbstractEvaluator.h"


class SolidAngleEvaluator : public AbstractEvaluator
{
public:
    SolidAngleEvaluator();
    virtual ~SolidAngleEvaluator();

    virtual double solidAngle(const glm::dvec3& a, const glm::dvec3& b, const glm::dvec3& c) const;

    virtual double tetQuality(const glm::dvec3 vp[]) const override;

    virtual double priQuality(const glm::dvec3 vp[]) const override;

    virtual double hexQuality(const glm::dvec3 vp[]) const override;
};

#endif // GPUMESH_SOLIDANGLEEVALUATOR
