#ifndef GPUMESH_METRICFREEMEASURER
#define GPUMESH_METRICFREEMEASURER

#include "AbstractMeasurer.h"


class MetricFreeMeasurer : public AbstractMeasurer
{
public:
    MetricFreeMeasurer();
    virtual ~MetricFreeMeasurer();


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
};

#endif // GPUMESH_METRICFREEMEASURER
