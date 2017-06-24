#include "ComputedTexSampler.h"

#include <CellarWorkbench/Misc/Log.h>

#include "LocalSampler.h"

using namespace cellar;


ComputedTexSampler::ComputedTexSampler() :
    TextureSampler("Computed Texture")
{
}

ComputedTexSampler::~ComputedTexSampler()
{
}

bool ComputedTexSampler::isMetricWise() const
{
    return true;
}

bool ComputedTexSampler::useComputedMetric() const
{
    return true;
}

void ComputedTexSampler::updateAnalyticalMetric(
        const Mesh& mesh)
{

}

void ComputedTexSampler::updateComputedMetric(
        const Mesh& mesh,
        const std::shared_ptr<LocalSampler>& sampler)
{
    buildGrid(mesh, *sampler);
}
