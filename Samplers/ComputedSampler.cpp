#include "ComputedSampler.h"

#include <CellarWorkbench/Misc/Log.h>

using namespace cellar;


ComputedSampler::ComputedSampler() :
    LocalSampler("Computed")
{
}

ComputedSampler::~ComputedSampler()
{
}

void ComputedSampler::setReferenceMesh(
        const Mesh& mesh)
{
    // Prevent application from updating background mesh
}

void ComputedSampler::setComputedMetrics(
    const Mesh& mesh,
    const std::vector<MeshMetric>& metrics)
{
    LocalSampler::setReferenceMesh(mesh);
    _refMetrics = metrics;
}
