#include "AbstractDiscretizer.h"

#include <GLM/gtc/constants.hpp>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


AbstractDiscretizer::AbstractDiscretizer(
        const std::string& name,
        const std::string& shader) :
    _discretizationName(name),
    _discretizationShader(shader),
    _baseShader(":/glsl/compute/Discretizing/Base.glsl")
{

}

AbstractDiscretizer::~AbstractDiscretizer()
{

}

std::string AbstractDiscretizer::discretizationShader() const
{
    return _discretizationShader;
}

void AbstractDiscretizer::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    if(!_discretizationShader.empty())
    {
        program.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            _baseShader.c_str()
        });

        program.addShader(GL_COMPUTE_SHADER, {
            mesh.meshGeometryShaderName(),
            _discretizationShader.c_str()
        });
    }
}

void AbstractDiscretizer::setPluginUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

void AbstractDiscretizer::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
}

Metric AbstractDiscretizer::interpolateMetrics(
        const Metric& m1,
        const Metric& m2,
        double a) const
{
    return glm::mix(m1, m2, a);
}

Metric AbstractDiscretizer::vertMetric(const Mesh& mesh, unsigned int vId) const
{
    return vertMetric(mesh.verts[vId].p);
}

Metric AbstractDiscretizer::vertMetric(const glm::dvec3& position) const
{
    glm::dvec3 vp = position * glm::dvec3(7);

    double localElemSize = 0.0;
    localElemSize = 1.0 / glm::pow(1000, 1.0/3.0);

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

void AbstractDiscretizer::tetrahedrizeMesh(const Mesh& mesh, std::vector<MeshTet>& tets)
{
    tets = mesh.tets;

    size_t priCount = mesh.pris.size();
    for(size_t p=0; p < priCount; ++p)
    {
        const MeshPri& pri = mesh.pris[p];
        for(uint t=0; t < MeshPri::TET_COUNT; ++t)
            tets.push_back(MeshTet(
                pri.v[MeshPri::tets[t][0]],
                pri.v[MeshPri::tets[t][1]],
                pri.v[MeshPri::tets[t][2]],
                pri.v[MeshPri::tets[t][3]]));
    }

    size_t hexCount = mesh.hexs.size();
    for(size_t h=0; h < hexCount; ++h)
    {
        const MeshHex& hex = mesh.hexs[h];
        for(uint t=0; t < MeshHex::TET_COUNT; ++t)
            tets.push_back(MeshTet(
                hex.v[MeshHex::tets[t][0]],
                hex.v[MeshHex::tets[t][1]],
                hex.v[MeshHex::tets[t][2]],
                hex.v[MeshHex::tets[t][3]]));
    }
}
