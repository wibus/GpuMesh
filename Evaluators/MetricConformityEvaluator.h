#ifndef GPUMESH_METRICCONFORMITYEVALUATOR
#define GPUMESH_METRICCONFORMITYEVALUATOR

#include "AbstractEvaluator.h"

#include "Samplers/AbstractSampler.h"


class MetricConformityEvaluator : public AbstractEvaluator
{
public:
    MetricConformityEvaluator();
    virtual ~MetricConformityEvaluator();

    virtual double tetQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshTet& tet) const override;

    virtual double priQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshPri& pri) const override;

    virtual double hexQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshHex& hex) const override;

protected:
    Metric specifiedMetric(const AbstractSampler& sampler,
            const glm::dvec3& v0,
            const glm::dvec3& v1,
            const glm::dvec3& v2,
            const glm::dvec3& v3,
            uint vId) const;
    double metricConformity(
            const glm::dmat3& Fk,
            const Metric& avrgMetric) const;
};

#endif // GPUMESH_METRICCONFORMITYEVALUATOR
