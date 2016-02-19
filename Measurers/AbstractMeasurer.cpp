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

void AbstractMeasurer::setPluginUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

void AbstractMeasurer::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{

}

double AbstractMeasurer::tetVolume(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const MeshTet& tet) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]],
        mesh.verts[tet.v[1]],
        mesh.verts[tet.v[2]],
        mesh.verts[tet.v[3]],
    };

    return tetVolume(sampler, vp);
}

double AbstractMeasurer::priVolume(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const MeshPri& pri) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[pri.v[0]],
        mesh.verts[pri.v[1]],
        mesh.verts[pri.v[2]],
        mesh.verts[pri.v[3]],
        mesh.verts[pri.v[4]],
        mesh.verts[pri.v[5]]
    };

    return priVolume(sampler, vp);
}

double AbstractMeasurer::hexVolume(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const MeshHex& hex) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[hex.v[0]],
        mesh.verts[hex.v[1]],
        mesh.verts[hex.v[2]],
        mesh.verts[hex.v[3]],
        mesh.verts[hex.v[4]],
        mesh.verts[hex.v[5]],
        mesh.verts[hex.v[6]],
        mesh.verts[hex.v[7]]
    };

    return hexVolume(sampler, vp);
}

double AbstractMeasurer::computeLocalElementSize(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        size_t vId) const
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
