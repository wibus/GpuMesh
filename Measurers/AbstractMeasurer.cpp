#include "AbstractMeasurer.h"

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Discretizers/AbstractDiscretizer.h"

using namespace std;


AbstractMeasurer::AbstractMeasurer(
        const string& name,
        const string& shader) :
    _measureName(name),
    _measureShader(shader)
{

}

AbstractMeasurer::~AbstractMeasurer()
{

}

std::string AbstractMeasurer::measureShader() const
{
    return _measureShader;
}

void AbstractMeasurer::installPlugIn(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/Base.glsl"
    });

    program.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        _measureShader.c_str()
    });
}

void AbstractMeasurer::uploadUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

double AbstractMeasurer::tetVolume(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        const MeshTet& tet) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]],
        mesh.verts[tet.v[1]],
        mesh.verts[tet.v[2]],
        mesh.verts[tet.v[3]],
    };

    return tetVolume(discretizer, vp);
}

double AbstractMeasurer::priVolume(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
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

    return priVolume(discretizer, vp);
}

double AbstractMeasurer::hexVolume(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
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

    return hexVolume(discretizer, vp);
}

double AbstractMeasurer::computeLocalElementSize(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
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
