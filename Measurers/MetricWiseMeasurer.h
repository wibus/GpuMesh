#ifndef GPUMESH_METRICWISEMEASURER
#define GPUMESH_METRICWISEMEASURER

#include "AbstractMeasurer.h"


class MetricWiseMeasurer : public AbstractMeasurer
{
public:
    MetricWiseMeasurer();
    virtual ~MetricWiseMeasurer();


    // Distances
    virtual double riemannianDistance(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& a,
            const glm::dvec3& b) const override;

    virtual glm::dvec3 riemannianSegment(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& a,
            const glm::dvec3& b) const override;


    // Volumes
    using AbstractMeasurer::tetVolume;
    using AbstractMeasurer::priVolume;
    using AbstractMeasurer::hexVolume;

    virtual double tetVolume(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3 vp[]) const override;

    virtual double priVolume(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3 vp[]) const override;

    virtual double hexVolume(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3 vp[]) const override;


    // High level measurements
    virtual glm::dvec3 computeVertexEquilibrium(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            size_t vId) const override;

protected:
    virtual glm::dvec3 computeSpringForce(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& pi,
            const glm::dvec3& pj) const;
};

#endif // GPUMESH_METRICWISEMEASURER