#include "LocalSampler.h"

#include <CellarWorkbench/Misc/Log.h>

#include <DataStructures/Mesh.h>
#include <DataStructures/TriSet.h>

using namespace cellar;

LocalTet::LocalTet(const MeshTet& t)
    { v[0] = t.v[0]; v[1] = t.v[1]; v[2] = t.v[2]; v[3] = t.v[3];
      n[0] = -1;     n[1] = -1;     n[2] = -1;     n[3] = -1;     }


// CUDA Drivers Interface
void installCudaLocalSampler();


LocalSampler::LocalSampler() :
    AbstractSampler("Local", ":/glsl/compute/Sampling/Local.glsl", installCudaLocalSampler),
    _debugMesh(new Mesh())
{
    _debugMesh->modelName = "Local sampling mesh";
}

LocalSampler::~LocalSampler()
{
}

bool LocalSampler::isMetricWise() const
{
    return true;
}

void LocalSampler::initialize()
{
}

void LocalSampler::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
}

void LocalSampler::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
}

void LocalSampler::setReferenceMesh(
        const Mesh& mesh,
        int density)
{
    // Clear resources
    _debugMesh->clear();

    _vertMetrics.clear();
    _vertMetrics.shrink_to_fit();
    _localTets.clear();
    _localTets.shrink_to_fit();
    _indexCache.clear();
    _indexCache.shrink_to_fit();


    // Break prisms and hex into tetrahedra
    tetrahedrizeMesh(_localTets, mesh);
    size_t tetCount = _localTets.size();
    size_t triCount = tetCount * 4;
    if(tetCount == 0)
    {
        getLog().postMessage(new Message('I', false,
            "Empty refrence mesh : no local tets created", "LocalSampler"));
        return;
    }


    // Find tets neighbors
    TriSet triSet;
    size_t bucketCount = size_t(pow(double(tetCount), 1/3.0));
    triSet.reset(bucketCount);

    getLog().postMessage(new Message('I', false,
        "Finding local tets neighborhood (bucket="+
        std::to_string(bucketCount)+")", "LocalSampler"));

    for(size_t t=0; t < tetCount; ++t)
    {
        LocalTet& tet = _localTets[t];
        for(uint s=0; s < MeshTet::TRI_COUNT; ++s)
        {
            Triangle tri(tet.v[MeshTet::tris[s][0]],
                         tet.v[MeshTet::tris[s][1]],
                         tet.v[MeshTet::tris[s][2]]);
            glm::uvec2 con = triSet.xOrTri(tri, t, s);

            uint owner = con[0];
            uint side = con[1];
            tet.n[s] = owner;

            if(owner != TriSet::NO_OWNER)
            {
                _localTets[owner].n[side] = t;
            }
        }
    }

    const std::vector<Triangle>& surfTri = triSet.gather();
    size_t remTriCount = surfTri.size();

    for(const Triangle& tri : surfTri)
    {
        size_t vertBase = _debugMesh->verts.size();
        _debugMesh->verts.push_back(mesh.verts[tri.v[0]]);
        _debugMesh->verts.push_back(mesh.verts[tri.v[1]]);
        _debugMesh->verts.push_back(mesh.verts[tri.v[2]]);
        _debugMesh->verts.push_back((mesh.verts[tri.v[0]].p +
            mesh.verts[tri.v[1]].p + mesh.verts[tri.v[2]].p)/3.0);

        _debugMesh->tets.push_back(MeshTet(vertBase, vertBase+1, vertBase+2, vertBase+3));
    }


    getLog().postMessage(new Message('I', false,
        "Surface triangle count: " + std::to_string(remTriCount) +
        " / " + std::to_string(triCount), "LocalSampler"));
    triSet.releaseMemoryPool();


    // Build index cache
}

Metric LocalSampler::metricAt(
        const glm::dvec3& position) const
{
    return vertMetric(position);
}

void LocalSampler::releaseDebugMesh()
{
}

const Mesh& LocalSampler::debugMesh()
{
    return *_debugMesh;
}

void LocalSampler::tetrahedrizeMesh(
        std::vector<LocalTet>& tets,
        const Mesh& mesh)
{
    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t maxTetCount = tetCount +
        priCount * 3 + hexCount * 6;

    tets.reserve(maxTetCount);

    for(const MeshTet& tet : mesh.tets)
        tets.push_back(tet);
}
