#ifndef GPUMESH_MEANRATIOEVALUATOR
#define GPUMESH_MEANRATIOEVALUATOR

#include "AbstractEvaluator.h"


class MeanRatioEvaluator : public AbstractEvaluator
{
public:
    MeanRatioEvaluator();
    virtual ~MeanRatioEvaluator();

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
    virtual double cornerQuality(const glm::dmat3& Fk) const;
};

#endif // GPUMESH_MEANRATIOEVALUATOR
