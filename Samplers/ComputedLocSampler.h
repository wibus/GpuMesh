#ifndef GPUMESH_COMPUTEDLOCSAMPLER
#define GPUMESH_COMPUTEDLOCSAMPLER

#include <GL3/gl3w.h>

#include "AbstractSampler.h"

class LocalSampler;


class ComputedLocSampler : public AbstractSampler
{
public:
    ComputedLocSampler();
    virtual ~ComputedLocSampler();


    virtual bool isMetricWise() const override;

    virtual bool useComputedMetric() const override;


    virtual void updateGlslData(const Mesh& mesh) const override;

    virtual void updateCudaData(const Mesh& mesh) const override;

    virtual void clearGlslMemory(const Mesh& mesh) const override;

    virtual void clearCudaMemory(const Mesh& mesh) const override;


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
    std::shared_ptr<LocalSampler> _localSampler;
};

#endif // GPUMESH_COMPUTEDLOCSAMPLER
