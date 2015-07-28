#ifndef GPUMESH_MEANRATIOEVALUATOR
#define GPUMESH_MEANRATIOEVALUATOR

#include "AbstractEvaluator.h"


class MeanRatioEvaluator : public AbstractEvaluator
{
public:
    MeanRatioEvaluator();
    virtual ~MeanRatioEvaluator();

    virtual double tetQuality(const glm::dvec3 vp[]) const override;

    virtual double priQuality(const glm::dvec3 vp[]) const override;

    virtual double hexQuality(const glm::dvec3 vp[]) const override;

protected:
    virtual double cornerQuality(const glm::dmat3& Fk) const;
};

#endif // GPUMESH_MEANRATIOEVALUATOR
