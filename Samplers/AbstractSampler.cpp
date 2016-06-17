#include "AbstractSampler.h"

#include <GLM/gtc/constants.hpp>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


void setCudaMetricScaling(double scaling);

AbstractSampler::AbstractSampler(
        const std::string& name,
        const std::string& shader,
        const installCudaFct installCuda) :
    _scaling(1.0),
    _samplingName(name),
    _samplingShader(shader),
    _baseShader(":/glsl/compute/Sampling/Base.glsl"),
    _installCuda(installCuda)
{

}

AbstractSampler::~AbstractSampler()
{

}

void AbstractSampler::initialize()
{

}

void AbstractSampler::setScaling(double scaling)
{
    _scaling = scaling;
}

std::string AbstractSampler::samplingShader() const
{
    return _samplingShader;
}

void AbstractSampler::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    if(!_samplingShader.empty())
    {
        program.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            _baseShader.c_str()
        });

        program.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            _samplingShader.c_str()
        });
    }

    _installCuda();
}

void AbstractSampler::setPluginGlslUniforms(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
    program.setFloat("MetricScaling", scaling());
}

void AbstractSampler::setPluginCudaUniforms(
        const Mesh& mesh) const
{
    setCudaMetricScaling(scaling());
}

Metric AbstractSampler::interpolateMetrics(
        const Metric& m1,
        const Metric& m2,
        double a) const
{
    return glm::mix(m1, m2, a);
}

Metric AbstractSampler::vertMetric(const Mesh& mesh, unsigned int vId) const
{
    return vertMetric(mesh.verts[vId].p);
}

Metric AbstractSampler::vertMetric(const glm::dvec3& position) const
{
    glm::dvec3 vp = position * glm::dvec3(7);

    double localElemSize = 0.0;
    localElemSize = 1.0 / scaling();

    double elemSize = localElemSize;
    double elemSizeInv2 = 1.0 / (elemSize * elemSize);

    double scale = glm::pow(3, glm::sin(vp.x));
    double targetElemSizeX = elemSize * scale;
    double targetElemSizeXInv2 = 1.0 / (targetElemSizeX * targetElemSizeX);
    double targetElemSizeZ = elemSize / scale;
    double targetElemSizeZInv2 = 1.0 / (targetElemSizeZ * targetElemSizeZ);

    double rx = targetElemSizeXInv2;
    double ry = elemSizeInv2;
    double rz = elemSizeInv2;

    return Metric(
        glm::dvec3(rx, 0,  0),
        glm::dvec3(0,  ry, 0),
        glm::dvec3(0,  0,  rz));
}

void AbstractSampler::boundingBox(
        const Mesh& mesh,
        glm::dvec3& minBounds,
        glm::dvec3& maxBounds) const
{
    minBounds = glm::dvec3(INFINITY);
    maxBounds = glm::dvec3(-INFINITY);
    size_t vertCount = mesh.verts.size();
    for(size_t v=0; v < vertCount; ++v)
    {
        const glm::dvec3& vertPos = mesh.verts[v].p;
        minBounds = glm::min(minBounds, vertPos);
        maxBounds = glm::max(maxBounds, vertPos);
    }
}
