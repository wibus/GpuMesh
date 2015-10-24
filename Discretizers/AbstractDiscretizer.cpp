#include "AbstractDiscretizer.h"

#include <GLM/gtc/constants.hpp>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


AbstractDiscretizer::AbstractDiscretizer(
        const std::string& name,
        const std::string& shader) :
    _discretizationName(name),
    _discretizationShader(shader)
{

}

AbstractDiscretizer::~AbstractDiscretizer()
{

}

std::string AbstractDiscretizer::discretizationShader() const
{
    return _discretizationShader;
}

void AbstractDiscretizer::installPlugIn(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    if(!_discretizationShader.empty())
    {
        program.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            _discretizationShader.c_str()
        });
    }
}

void AbstractDiscretizer::uploadUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

Metric AbstractDiscretizer::interpolate(
        const Metric& m1,
        const Metric& m2,
        double a) const
{
    return glm::mix(m1, m2, a);
}

Metric AbstractDiscretizer::vertMetric(const Mesh& mesh, uint vId) const
{
    return vertMetric(mesh.verts[vId].p);
}

Metric AbstractDiscretizer::vertMetric(const glm::dvec3& pos) const
{
    glm::dvec3 vp = pos * glm::dvec3(7);

    double localElemSize = 0.0;
    localElemSize = 1.0 / glm::pow(1000, 1.0/3.0);

    double elemSize = localElemSize;
    double elemSizeInv2 = 1.0 / (elemSize * elemSize);

    double scale = glm::pow(4, glm::sin(vp.x));
    double targetElemSizeX = elemSize * scale;
    double targetElemSizeXInv2 = 1.0 / (targetElemSizeX * targetElemSizeX);
    double targetElemSizeZ = elemSize / scale;
    double targetElemSizeZInv2 = 1.0 / (targetElemSizeZ * targetElemSizeZ);

    double rx = targetElemSizeXInv2;
    double ry = targetElemSizeXInv2;
    double rz = targetElemSizeXInv2;

    return Metric(
        glm::dvec3(rx, 0,  0),
        glm::dvec3(0,  ry, 0),
        glm::dvec3(0,  0,  rz));
}

void AbstractDiscretizer::boundingBox(
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
