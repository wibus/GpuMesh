#ifndef GPUMESH_COMPUTEDTEXSAMPLER
#define GPUMESH_COMPUTEDTEXSAMPLER

#include <GL3/gl3w.h>

#include "TextureSampler.h"


class ComputedTexSampler : public TextureSampler
{
public:
    ComputedTexSampler();
    virtual ~ComputedTexSampler();


    virtual bool isMetricWise() const override;

    virtual bool useComputedMetric() const override;


    virtual void updateAnalyticalMetric(
            const Mesh& mesh) override;

    virtual void updateComputedMetric(
            const Mesh& mesh,
            const std::shared_ptr<LocalSampler>& sampler) override;
};

#endif // GPUMESH_COMPUTEDTEXSAMPLER
