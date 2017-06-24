#include "AnalyticSampler.h"

#include "DataStructures/Mesh.h"


// CUDA Drivers Interface
void installCudaAnalyticSampler();


AnalyticSampler::AnalyticSampler() :
    AbstractSampler("Analytic", ":/glsl/compute/Sampling/Analytic.glsl", installCudaAnalyticSampler),
    _debugMesh(new Mesh())
{
    _debugMesh->modelName = "Analytic sampling mesh";
}

AnalyticSampler::~AnalyticSampler()
{

}

bool AnalyticSampler::isMetricWise() const
{
    return true;
}

bool AnalyticSampler::useComputedMetric() const
{
    return false;
}

void AnalyticSampler::updateAnalyticalMetric(
        const Mesh& mesh)
{

}

void AnalyticSampler::updateComputedMetric(
        const Mesh& mesh,
        const std::shared_ptr<LocalSampler>& sampler)
{

}

MeshMetric AnalyticSampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    return vertMetric(position);
}

void AnalyticSampler::releaseDebugMesh()
{
}

const Mesh& AnalyticSampler::debugMesh()
{
    return *_debugMesh;
}
