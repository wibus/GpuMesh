#include "LocalSampler.h"

#include <array>

#include <CellarWorkbench/Misc/Log.h>

#include <DataStructures/GpuMesh.h>
#include <DataStructures/TriSet.h>
#include <DataStructures/Tetrahedralizer.h>

using namespace cellar;


// CUDA Drivers Interface
void installCudaLocalSampler();
void updateCudaLocalTets(
        const std::vector<GpuLocalTet>& localTetsBuff);
void updateCudaLocalCache(
        const std::vector<GLuint>& localCacheBuff);
void updateCudaRefVerts(
        const std::vector<GpuVert>& refVertsBuff);
void updateCudaRefMetrics(
        const std::vector<glm::mat4>& refMetricsBuff);


LocalSampler::LocalSampler() :
    AbstractSampler("Local", ":/glsl/compute/Sampling/Local.glsl", installCudaLocalSampler),
    _debugMesh(new Mesh()),
    _localTetsSsbo(0),
    _localCacheSsbo(0),
    _refVertsSsbo(0),
    _refMetricsSsbo(0),
    _metricAtSub(0)
{
    _debugMesh->modelName = "Local sampling mesh";
}

LocalSampler::~LocalSampler()
{
    glDeleteBuffers(1, &_localTetsSsbo);
    _localTetsSsbo = 0;
    glDeleteBuffers(1, &_localCacheSsbo);
    _localCacheSsbo = 0;
    glDeleteBuffers(1, &_refVertsSsbo);
    _refVertsSsbo = 0;
    glDeleteBuffers(1, &_refMetricsSsbo);
    _refMetricsSsbo = 0;
}

bool LocalSampler::isMetricWise() const
{
    return true;
}

void LocalSampler::initialize()
{
    if(_localTetsSsbo == 0)
        glGenBuffers(1, &_localTetsSsbo);

    if(_localCacheSsbo == 0)
        glGenBuffers(1, &_localCacheSsbo);

    if(_refVertsSsbo == 0)
        glGenBuffers(1, &_refVertsSsbo);

    if(_refMetricsSsbo == 0)
        glGenBuffers(1, &_refMetricsSsbo);
}

void LocalSampler::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractSampler::installPlugin(mesh, program);

    GLuint localTets  = mesh.bufferBinding(EBufferBinding::LOCAL_TETS_BUFFER_BINDING);
    GLuint localCache = mesh.bufferBinding(EBufferBinding::LOCAL_CACHE_BUFFER_BINDING);
    GLuint refVerts   = mesh.bufferBinding(EBufferBinding::REF_VERTS_BUFFER_BINDING);
    GLuint refMetrics = mesh.bufferBinding(EBufferBinding::REF_METRICS_BUFFER_BINDING);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, localTets,  _localTetsSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, localCache, _localCacheSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, refVerts,   _refVertsSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, refMetrics, _refMetricsSsbo);
}

void LocalSampler::setupPluginExecution(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
    AbstractSampler::setupPluginExecution(mesh, program);

    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &_metricAtSub);
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
    _localCache.resize(mesh.verts.size());
    _localCache.shrink_to_fit();
    _refVerts = mesh.verts;
    _refVerts.shrink_to_fit();
    _refMetrics.resize(mesh.verts.size());
    for(size_t vId=0; vId < vertCount; ++vId)
        _refMetrics[vId] = vertMetric(mesh, vId);
    _refMetrics.shrink_to_fit();


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
        _localCache[tet.v[0]] = t;
        _localCache[tet.v[1]] = t;
        _localCache[tet.v[2]] = t;
        _localCache[tet.v[3]] = t;
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




    std::vector<GpuLocalTet> gpuLocalTets;
    gpuLocalTets.reserve(_localTets.size());
    for(const auto& t : _localTets)
        gpuLocalTets.push_back(GpuLocalTet(t));

    updateCudaLocalTets(gpuLocalTets);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _localTetsSsbo);
    size_t localTetsSize = sizeof(decltype(gpuLocalTets.front())) * gpuLocalTets.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, localTetsSize, gpuLocalTets.data(), GL_STATIC_DRAW);
    gpuLocalTets.clear();
    gpuLocalTets.shrink_to_fit();


    std::vector<GLuint> gpuLocalCache;
    gpuLocalCache.reserve(_localCache.size());
    for(const auto& i : _localCache)
        gpuLocalCache.push_back(i);

    updateCudaLocalCache(gpuLocalCache);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _localCacheSsbo);
    size_t localCacheSize = sizeof(decltype(gpuLocalCache.front())) * gpuLocalCache.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, localCacheSize, gpuLocalCache.data(), GL_STATIC_DRAW);
    gpuLocalCache.clear();
    gpuLocalCache.shrink_to_fit();


    // Reference Mesh Vertices
    std::vector<GpuVert> gpuRefVerts;
    gpuRefVerts.reserve(_refVerts.size());
    for(const auto& v : _refVerts)
        gpuRefVerts.push_back(GpuVert(v));

    updateCudaRefVerts(gpuRefVerts);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refVertsSsbo);
    size_t refVertsSize = sizeof(decltype(gpuRefVerts.front())) * gpuRefVerts.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, refVertsSize, gpuRefVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // Reference Mesh Metrics
    std::vector<glm::mat4> gpuRefMetrics;
    gpuRefMetrics.reserve(_refMetrics.size());
    for(const auto& metric : _refMetrics)
        gpuRefMetrics.push_back(glm::mat4(metric));

    updateCudaRefMetrics(gpuRefMetrics);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refMetricsSsbo);
    size_t refMetricsSize = sizeof(decltype(gpuRefMetrics.front())) * gpuRefMetrics.size();
    glBufferData(GL_SHADER_STORAGE_BUFFER, refMetricsSize, gpuRefMetrics.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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
        uint cacheId) const
{
    // Taboo search structures
    size_t tabooCount = 0;
    uint taboo[MAX_TABOO];

    uint tetId = _localCache[cacheId];
    const LocalTet* tet = &_localTets[tetId];

    double coor[4];
    while(!tetParams(_refVerts, *tet, position, coor))
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
    _localCache[cacheId] = tetId;

    return coor[0] * _refMetrics[tet->v[0]] +
           coor[1] * _refMetrics[tet->v[1]] +
           coor[2] * _refMetrics[tet->v[2]] +
           coor[3] * _refMetrics[tet->v[3]];
}

void LocalSampler::releaseDebugMesh()
{
}

const Mesh& LocalSampler::debugMesh()
{
    return *_debugMesh;
}
