#include "UniformSampler.h"

#include "DataStructures/Mesh.h"


// CUDA Drivers Interface
void installCudaUniformSampler();


UniformSampler::UniformSampler() :
    AbstractSampler("Uniform", ":/glsl/compute/Sampling/Uniform.glsl", installCudaUniformSampler),
    _debugMesh(new Mesh())
{
    _debugMesh->modelName = "Uniform sampling mesh";
}

UniformSampler::~UniformSampler()
{

}

bool UniformSampler::isMetricWise() const
{
    return false;
}

void UniformSampler::setReferenceMesh(
        const Mesh& mesh)
{

}

Metric UniformSampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    // Constant isotropic metric
    return Metric(scaling() * scaling());
}

void UniformSampler::releaseDebugMesh()
{
    // Mesh is not big
    // Never release it
}

const Mesh& UniformSampler::debugMesh()
{
    return *_debugMesh;
}
