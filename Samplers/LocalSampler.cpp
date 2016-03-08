#include "LocalSampler.h"

#include <array>

#include <CellarWorkbench/Misc/Log.h>

#include <DataStructures/Mesh.h>
#include <DataStructures/TriSet.h>
#include <DataStructures/Tetrahedralizer.h>

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
    AbstractSampler::installPlugin(mesh, program);
}

void LocalSampler::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
    AbstractSampler::setupPluginExecution(mesh, program);
}

void LocalSampler::setReferenceMesh(
        const Mesh& mesh,
        int density)
{
    size_t vertCount = mesh.verts.size();

    // Clear resources
    _debugMesh->clear();

    _localTets.clear();
    _localTets.shrink_to_fit();
    _indexCache.resize(mesh.verts.size());
    _indexCache.shrink_to_fit();
    _localVerts = mesh.verts;
    _localVerts.shrink_to_fit();
    _vertMetrics.resize(mesh.verts.size());
    for(size_t vId=0; vId < vertCount; ++vId)
        _vertMetrics[vId] = vertMetric(mesh, vId);
    _vertMetrics.shrink_to_fit();


    // Break prisms and hex into tetrahedra
    tetrahedrize(_localTets, mesh);
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
    size_t bucketCount = pow(double(triCount), 2.0/3.0);
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

        // At the end, the cache will be initialized with
        // the last tetrahedron for witch the vertex is node
        _indexCache[tet.v[0]] = t;
        _indexCache[tet.v[1]] = t;
        _indexCache[tet.v[2]] = t;
        _indexCache[tet.v[3]] = t;
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
}

const size_t MAX_TABOO = 32;
bool isTaboo(uint tId, uint taboo[], size_t count)
{
    if(tId != -1)
    {
        for(size_t i=0; i < count; ++i)
            if(tId == taboo[i])
                return true;
    }

    return false;
}

Metric LocalSampler::metricAt(
        const glm::dvec3& position,
        uint vertOwnerId) const
{
    // Taboo search structures
    size_t tabooCount = 0;
    uint taboo[MAX_TABOO];

    uint tetId = _indexCache[vertOwnerId];
    const LocalTet* tet = &_localTets[tetId];

    double coor[4];
    while(!tetParams(_localVerts, *tet, position, coor))
    {
        uint n = -1;
        double minCoor = 1/0.0;

        if(coor[0] < minCoor && !isTaboo(tet->n[0], taboo, tabooCount))
        {
            n = 0;
            minCoor = coor[0];
        }
        if(coor[1] < minCoor && !isTaboo(tet->n[1], taboo, tabooCount))
        {
            n = 1;
            minCoor = coor[1];
        }
        if(coor[2] < minCoor && !isTaboo(tet->n[2], taboo, tabooCount))
        {
            n = 2;
            minCoor = coor[2];
        }
        if(coor[3] < minCoor && !isTaboo(tet->n[3], taboo, tabooCount))
        {
            n = 3;
            minCoor = coor[3];
        }

        bool clipCurrentTet = false;
        if(n != -1)
        {
            uint nextTet = tet->n[n];

            if((nextTet != -1))
            {
                if(tabooCount < MAX_TABOO)
                {
                    // Add last tet to taboo list
                    taboo[tabooCount] = tetId;
                    ++tabooCount;

                    // Fetch the next local tet
                    tet = &_localTets[nextTet];
                    tetId = nextTet;
                }
                else
                {
                    // We went too far,
                    // stay where we are
                    clipCurrentTet = true;
                    getLog().postMessage(new Message('E', false,
                       "Visited to many tets during local search", "LocalSampler"));
                }
            }
            else
            {
                // The sampling point is on
                // the other side of the boundary
                clipCurrentTet = true;
                // This may not be an issue
            }
        }
        else
        {
            // Every surrounding tet
            // were already visited
            clipCurrentTet = true;
            getLog().postMessage(new Message('E', false,
               "Surrounded by taboo tets during local search", "LocalSampler"));
        }


        if(clipCurrentTet)
        {
            // Clamp sample to current tet
            // It's seems to be the closest
            // we can get to the sampling point
            double sum = 0.0;
            if(coor[0] < 0.0) coor[0] = 0.0; else sum += coor[0];
            if(coor[1] < 0.0) coor[1] = 0.0; else sum += coor[1];
            if(coor[2] < 0.0) coor[2] = 0.0; else sum += coor[2];
            if(coor[3] < 0.0) coor[3] = 0.0; else sum += coor[3];
            coor[0] /= sum;
            coor[1] /= sum;
            coor[2] /= sum;
            coor[3] /= sum;
            break;
        }
    }

    // TODO wbussiere 2016-03-07 :
    //  Verify potential race conditions issues
    _indexCache[vertOwnerId] = tetId;

    return coor[0] * _vertMetrics[tet->v[0]] +
           coor[1] * _vertMetrics[tet->v[1]] +
           coor[2] * _vertMetrics[tet->v[2]] +
           coor[3] * _vertMetrics[tet->v[3]];
}

void LocalSampler::releaseDebugMesh()
{
}

const Mesh& LocalSampler::debugMesh()
{
    return *_debugMesh;
}
