#ifndef GPUMESH_UNIFORMSAMPLER
#define GPUMESH_UNIFORMSAMPLER

#include "AbstractSampler.h"


class UniformSampler : public AbstractSampler
{
public:
    UniformSampler();
    virtual ~UniformSampler();


    virtual bool isMetricWise() const override;

    virtual bool useComputedMetric() const override;


    virtual void updateAnalyticalMetric(
            const Mesh& mesh) override;

    virtual void updateComputedMetric(
            const Mesh& mesh,
            const std::shared_ptr<LocalSampler>& sampler) override;


    virtual MeshMetric metricAt(
            const glm::dvec3& position,
            uint& cachedRefTet) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


private:
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_UNIFORMSAMPLER
