#ifndef GPUMESH_INSPHEREEDGEEVALUATOR
#define GPUMESH_INSPHEREEDGEEVALUATOR

#include "AbstractEvaluator.h"


class InsphereEdgeEvaluator : public AbstractEvaluator
{
public:
    InsphereEdgeEvaluator();
    virtual ~InsphereEdgeEvaluator();

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
};

#endif // GPUMESH_INSPHEREEDGEEVALUATOR
