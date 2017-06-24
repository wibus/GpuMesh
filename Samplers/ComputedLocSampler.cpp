#include "ComputedLocSampler.h"

#include <CellarWorkbench/Misc/Log.h>

#include "LocalSampler.h"

using namespace cellar;


// CUDA Drivers Interface
void installCudaLocalSampler();


ComputedLocSampler::ComputedLocSampler() :
    AbstractSampler("ComputedLoc", ":/glsl/compute/Sampling/Local.glsl", installCudaLocalSampler)
{
}

ComputedLocSampler::~ComputedLocSampler()
{
}

bool ComputedLocSampler::isMetricWise() const
{
    return true;
}

bool ComputedLocSampler::useComputedMetric() const
{
    return true;
}

void ComputedLocSampler::updateGlslData(const Mesh& mesh) const
{
    _localSampler->updateGlslData(mesh);
}

void ComputedLocSampler::updateCudaData(const Mesh& mesh) const
{
    _localSampler->updateCudaData(mesh);
}

void ComputedLocSampler::clearGlslMemory(const Mesh& mesh) const
{
    _localSampler->clearGlslMemory(mesh);
}

void ComputedLocSampler::clearCudaMemory(const Mesh& mesh) const
{
    _localSampler->clearCudaMemory(mesh);
}

void ComputedLocSampler::updateAnalyticalMetric(
        const Mesh& mesh)
{
}

void ComputedLocSampler::updateComputedMetric(
        const Mesh& mesh,
        const std::shared_ptr<LocalSampler>& sampler)
{
    _localSampler = sampler;
}

MeshMetric ComputedLocSampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    return _localSampler->metricAt(position, cachedRefTet);
}

void ComputedLocSampler::releaseDebugMesh()
{
    _localSampler->releaseDebugMesh();
}

const Mesh& ComputedLocSampler::debugMesh()
{
    return _localSampler->debugMesh();
}
