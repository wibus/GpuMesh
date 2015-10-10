#ifndef GPUMESH_METRICWISEMEASURER
#define GPUMESH_METRICWISEMEASURER

#include "AbstractMeasurer.h"


class MetricWiseMeasurer : public AbstractMeasurer
{
public:
    MetricWiseMeasurer();
    virtual ~MetricWiseMeasurer();


    virtual double computeLocalElementSize(
            const Mesh& mesh,
            size_t vId) const override;

    virtual glm::dvec3 computeVertexEquilibrium(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            size_t vId) const override;

    virtual double computePatchQuality(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            size_t vId) const override;

protected:
    virtual glm::dvec3 computeSpringForce(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& pi,
            const glm::dvec3& pj) const;
};

#endif // GPUMESH_METRICWISEMEASURER
