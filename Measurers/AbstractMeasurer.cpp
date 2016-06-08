#include "AbstractMeasurer.h"

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Samplers/AbstractSampler.h"

using namespace std;


AbstractMeasurer::AbstractMeasurer(
        const string& name,
        const string& shader,
        const installCudaFct installCuda) :
    _measureName(name),
    _measureShader(shader),
    _installCuda(installCuda)
{

}

AbstractMeasurer::~AbstractMeasurer()
{

}

void AbstractMeasurer::initialize()
{

}

std::string AbstractMeasurer::measureShader() const
{
    return _measureShader;
}

void AbstractMeasurer::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Measuring/Base.glsl"
    });

    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _measureShader.c_str()
    });

    _installCuda();
}

void AbstractMeasurer::setPluginGlslUniforms(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{

}

void AbstractMeasurer::setPluginCudaUniforms(
        const Mesh& mesh) const
{

}

double AbstractMeasurer::tetEuclideanVolume(
        const Mesh& mesh,
        const MeshTet& tet)
{
    glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]].p,
        mesh.verts[tet.v[1]].p,
        mesh.verts[tet.v[2]].p,
        mesh.verts[tet.v[3]].p,
    };

    return tetEuclideanVolume(vp);
}

double AbstractMeasurer::tetEuclideanVolume(
        const glm::dvec3 vp[])
{
    double detSum = glm::determinant(glm::dmat3(
        vp[3] - vp[0],
        vp[3] - vp[1],
        vp[3] - vp[2]));

    return detSum / 6.0;
}

double AbstractMeasurer::tetVolume(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const MeshTet& tet) const
{
    glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]].p,
        mesh.verts[tet.v[1]].p,
        mesh.verts[tet.v[2]].p,
        mesh.verts[tet.v[3]].p,
    };

    return tetVolume(sampler, vp, tet);
}

double AbstractMeasurer::priVolume(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const MeshPri& pri) const
{
    glm::dvec3 vp[] = {
        mesh.verts[pri.v[0]].p,
        mesh.verts[pri.v[1]].p,
        mesh.verts[pri.v[2]].p,
        mesh.verts[pri.v[3]].p,
        mesh.verts[pri.v[4]].p,
        mesh.verts[pri.v[5]].p,
    };

    return priVolume(sampler, vp, pri);
}

double AbstractMeasurer::hexVolume(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const MeshHex& hex) const
{
    glm::dvec3 vp[] = {
        mesh.verts[hex.v[0]].p,
        mesh.verts[hex.v[1]].p,
        mesh.verts[hex.v[2]].p,
        mesh.verts[hex.v[3]].p,
        mesh.verts[hex.v[4]].p,
        mesh.verts[hex.v[5]].p,
        mesh.verts[hex.v[6]].p,
        mesh.verts[hex.v[7]].p,
    };

    return hexVolume(sampler, vp, hex);
}

double AbstractMeasurer::computeLocalElementSize(
        const Mesh& mesh,
        uint vId) const
{
    const std::vector<MeshVert>& verts = mesh.verts;

    const glm::dvec3& pos = verts[vId].p;
    const MeshTopo& topo = mesh.topos[vId];
    const vector<MeshNeigVert>& neigVerts = topo.neighborVerts;

    double totalSize = 0.0;
    size_t neigVertCount = neigVerts.size();
    for(size_t n=0; n < neigVertCount; ++n)
    {
        totalSize += glm::length(pos - verts[neigVerts[n].v].p);
    }

    return totalSize / neigVertCount;
}
