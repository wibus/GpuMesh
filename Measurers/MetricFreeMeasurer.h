#ifndef GPUMESH_METRICFREEMEASURER
#define GPUMESH_METRICFREEMEASURER

#include "AbstractMeasurer.h"


class MetricFreeMeasurer : public AbstractMeasurer
{
public:
    MetricFreeMeasurer();
    virtual ~MetricFreeMeasurer();


    // Distances
    virtual double riemannianDistance(
            const AbstractSampler& sampler,
            const glm::dvec3& a,
            const glm::dvec3& b,
            uint& cachedRefTet) const override;

    virtual glm::dvec3 riemannianSegment(
            const AbstractSampler& sampler,
            const glm::dvec3& a,
            const glm::dvec3& b,
            uint& cachedRefTet) const override;


    // Volumes
    virtual double tetVolume(
            const AbstractSampler& sampler,
            const glm::dvec3 vp[],
            const MeshTet& tet) const override;

    virtual double priVolume(
            const AbstractSampler& sampler,
            const glm::dvec3 vp[],
            const MeshPri& pri) const override;

    virtual double hexVolume(
            const AbstractSampler& sampler,
            const glm::dvec3 vp[],
            const MeshHex& hex) const override;


    // High level measurements
    virtual glm::dvec3 computeVertexEquilibrium(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            uint vId) const override;

protected:
};

#endif // GPUMESH_METRICFREEMEASURER
