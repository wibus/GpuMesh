#ifndef GPUMESH_METRICCONFORMITYEVALUATOR
#define GPUMESH_METRICCONFORMITYEVALUATOR

#include "AbstractEvaluator.h"

#include "Discretizers/AbstractDiscretizer.h"


class MetricConformityEvaluator : public AbstractEvaluator
{
public:
    MetricConformityEvaluator();
    virtual ~MetricConformityEvaluator();

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
    Metric specifiedMetric(const AbstractDiscretizer& discretizer,
            const glm::dvec3& v0,
            const glm::dvec3& v1,
            const glm::dvec3& v2,
            const glm::dvec3& v3) const;
    double metricConformity(
            const glm::dmat3& Fk,
            const Metric& avrgMetric) const;
};

#endif // GPUMESH_METRICCONFORMITYEVALUATOR