#include "DummySampler.h"

#include "DataStructures/Mesh.h"


// CUDA Drivers Interface
void installCudaDummySampler();


DummySampler::DummySampler() :
    AbstractSampler("Dummy", ":/glsl/compute/Sampling/Dummy.glsl", installCudaDummySampler),
    _debugMesh(new Mesh())
{
    _debugMesh->modelName = "Dummy sampling mesh";
}

DummySampler::~DummySampler()
{

}

bool DummySampler::isMetricWise() const
{
    return false;
}

void DummySampler::setReferenceMesh(
        const Mesh& mesh)
{

}

Metric DummySampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    // Constant isotropic metric
    return Metric(1.0);
}

void DummySampler::releaseDebugMesh()
{
    // Mesh is not big
    // Never release it
}

const Mesh& DummySampler::debugMesh()
{
    return *_debugMesh;
}
