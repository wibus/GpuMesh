#ifndef GPUMESH_ANALYTICSAMPLER
#define GPUMESH_ANALYTICSAMPLER

#include "AbstractSampler.h"


class AnalyticSampler : public AbstractSampler
{
public:
    AnalyticSampler();
    virtual ~AnalyticSampler();


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


protected:


private:
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_ANALYTICSAMPLER
