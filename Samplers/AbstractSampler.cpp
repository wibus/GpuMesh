#include "AbstractSampler.h"

#include <GLM/gtc/constants.hpp>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


AbstractSampler::AbstractSampler(
        const std::string& name,
        const std::string& shader,
        const installCudaFct installCuda) :
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

void AbstractSampler::setPluginUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

void AbstractSampler::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
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
    localElemSize = 1.0 / glm::pow(10000, 1.0/3.0);

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

bool AbstractSampler::tetParams(
        const std::vector<MeshVert>& verts,
        const MeshTet& tet,
        const glm::dvec3& p,
        double coor[4])
{
    // ref : https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Barycentric_coordinates_on_tetrahedra

    const glm::dvec3& vp0 = verts[tet.v[0]].p;
    const glm::dvec3& vp1 = verts[tet.v[1]].p;
    const glm::dvec3& vp2 = verts[tet.v[2]].p;
    const glm::dvec3& vp3 = verts[tet.v[3]].p;

    glm::dmat3 T(vp0 - vp3, vp1 - vp3, vp2 - vp3);

    glm::dvec3 y = glm::inverse(T) * (p - vp3);
    coor[0] = y[0];
    coor[1] = y[1];
    coor[2] = y[2];
    coor[3] = 1.0 - (y[0] + y[1] + y[2]);

    const double EPSILON_IN = -1e-8;
    bool isIn = (coor[0] >= EPSILON_IN && coor[1] >= EPSILON_IN &&
                 coor[2] >= EPSILON_IN && coor[3] >= EPSILON_IN);
    return isIn;
}
