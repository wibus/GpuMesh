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

void AnalyticSampler::setReferenceMesh(
        const Mesh& mesh,
        int density)
{

}

Metric AnalyticSampler::metricAt(
        const glm::dvec3& position,
        uint cacheId) const
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
