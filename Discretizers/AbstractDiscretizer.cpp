#include "AbstractDiscretizer.h"

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
    glm::dvec3 vp = mesh.verts[vId].p * glm::dvec3(10, 1, 1.0);

    double elemSize = 3.0 / glm::pow((double)mesh.verts.size(), 1/3.0);
    double elemSizeInv2 = 1.0 / (elemSize * elemSize);


    double scaleFactor = 3.0;
    double invScaleFactor = 1.0 / scaleFactor;

    double sinScale = (glm::sin(vp.x)+1)/2;
    double scale = (sinScale*sinScale)*(scaleFactor-invScaleFactor) + invScaleFactor;
    double targetElemSize = elemSize * scale;
    double targetElemSizeInv2 = 1.0 / (targetElemSize * targetElemSize);

    double rx = targetElemSizeInv2;
    double ry = elemSizeInv2;
    double rz = elemSizeInv2;

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
