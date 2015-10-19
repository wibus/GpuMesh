#ifndef GPUMESH_SOLIDANGLEEVALUATOR
#define GPUMESH_SOLIDANGLEEVALUATOR

#include "AbstractEvaluator.h"


class SolidAngleEvaluator : public AbstractEvaluator
{
public:
    SolidAngleEvaluator();
    virtual ~SolidAngleEvaluator();

    using AbstractEvaluator::tetQuality;
    using AbstractEvaluator::priQuality;
    using AbstractEvaluator::hexQuality;

    virtual double tetQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const override;

    virtual double priQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const override;

    virtual double hexQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const override;

protected:
    virtual double solidAngle(
            const glm::dvec3& a,
            const glm::dvec3& b,
            const glm::dvec3& c) const;
};

#endif // GPUMESH_SOLIDANGLEEVALUATOR
