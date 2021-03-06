#include "LocalSampler.h"

#include <array>

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/GL/GlProgram.h>

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
        const std::vector<GpuMetric>& refMetricsBuff);


LocalSampler::LocalSampler(const std::string& name) :
    AbstractSampler(name, ":/glsl/compute/Sampling/Local.glsl", installCudaLocalSampler),
    _debugMesh(nullptr),
    _localTetsSsbo(0),
    _refVertsSsbo(0),
    _refMetricsSsbo(0)
{
}

LocalSampler::LocalSampler() :
    AbstractSampler("Local", ":/glsl/compute/Sampling/Local.glsl", installCudaLocalSampler),
    _debugMesh(nullptr),
    _localTetsSsbo(0),
    _refVertsSsbo(0),
    _refMetricsSsbo(0)
{
}

LocalSampler::~LocalSampler()
{
    glDeleteBuffers(1, &_localTetsSsbo);
    _localTetsSsbo = 0;
    glDeleteBuffers(1, &_refVertsSsbo);
    _refVertsSsbo = 0;
    glDeleteBuffers(1, &_refMetricsSsbo);
    _refMetricsSsbo = 0;
}

bool LocalSampler::isMetricWise() const
{
    return true;
}

bool LocalSampler::useComputedMetric() const
{
    return false;
}

void LocalSampler::updateGlslData(const Mesh& mesh) const
{
    if(_localTetsSsbo == 0)
        glGenBuffers(1, &_localTetsSsbo);

    if(_refVertsSsbo == 0)
        glGenBuffers(1, &_refVertsSsbo);

    if(_refMetricsSsbo == 0)
        glGenBuffers(1, &_refMetricsSsbo);

    GLuint localTets  = mesh.glBufferBinding(EBufferBinding::LOCAL_TETS_BUFFER_BINDING);
    GLuint refVerts   = mesh.glBufferBinding(EBufferBinding::REF_VERTS_BUFFER_BINDING);
    GLuint refMetrics = mesh.glBufferBinding(EBufferBinding::REF_METRICS_BUFFER_BINDING);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, localTets,  _localTetsSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, refVerts,   _refVertsSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, refMetrics, _refMetricsSsbo);


    // Reference Mesh Tetrahedra
    {
        std::vector<GpuLocalTet> gpuLocalTets;
        gpuLocalTets.reserve(_localTets.size());
        for(const auto& t : _localTets)
            gpuLocalTets.push_back(GpuLocalTet(t.v, t.n));

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _localTetsSsbo);
        size_t localTetsSize = sizeof(decltype(gpuLocalTets.front())) * gpuLocalTets.size();
        glBufferData(GL_SHADER_STORAGE_BUFFER, localTetsSize, gpuLocalTets.data(), GL_STREAM_COPY);
    }


    // Reference Mesh Vertices
    {
        std::vector<GpuVert> gpuRefVerts;
        gpuRefVerts.reserve(_refVerts.size());
        for(const auto& v : _refVerts)
            gpuRefVerts.push_back(GpuVert(v));

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refVertsSsbo);
        size_t refVertsSize = sizeof(decltype(gpuRefVerts.front())) * gpuRefVerts.size();
        glBufferData(GL_SHADER_STORAGE_BUFFER, refVertsSize, gpuRefVerts.data(), GL_STREAM_COPY);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }


    // Reference Mesh Metrics
    {
        std::vector<GpuMetric> gpuRefMetrics;
        gpuRefMetrics.reserve(_refMetrics.size());
        for(const auto& metric : _refMetrics)
            gpuRefMetrics.push_back(GpuMetric(metric));

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refMetricsSsbo);
        size_t refMetricsSize = sizeof(decltype(gpuRefMetrics.front())) * gpuRefMetrics.size();
        glBufferData(GL_SHADER_STORAGE_BUFFER, refMetricsSize, gpuRefMetrics.data(), GL_STREAM_COPY);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
}

void LocalSampler::updateCudaData(const Mesh& mesh) const
{
    // Reference Mesh Tetrahedra
    {
        std::vector<GpuLocalTet> gpuLocalTets;
        gpuLocalTets.reserve(_localTets.size());
        for(const auto& t : _localTets)
            gpuLocalTets.push_back(GpuLocalTet(t.v, t.n));

        updateCudaLocalTets(gpuLocalTets);
    }


    // Reference Mesh Vertices
    {
        std::vector<GpuVert> gpuRefVerts;
        gpuRefVerts.reserve(_refVerts.size());
        for(const auto& v : _refVerts)
            gpuRefVerts.push_back(GpuVert(v));

        updateCudaRefVerts(gpuRefVerts);
    }


    // Reference Mesh Metrics
    {
        std::vector<GpuMetric> gpuRefMetrics;
        gpuRefMetrics.reserve(_refMetrics.size());
        for(const auto& metric : _refMetrics)
            gpuRefMetrics.push_back(GpuMetric(metric));

        updateCudaRefMetrics(gpuRefMetrics);
    }
}

void LocalSampler::clearGlslMemory(const Mesh& mesh) const
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _localTetsSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STREAM_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refVertsSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STREAM_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _refMetricsSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_STREAM_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void LocalSampler::clearCudaMemory(const Mesh& mesh) const
{
    // Reference Mesh Tetrahedra
    {
        std::vector<GpuLocalTet> gpuLocalTets;
        updateCudaLocalTets(gpuLocalTets);
    }

    // Reference Mesh Vertices
    {
        std::vector<GpuVert> gpuRefVerts;
        updateCudaRefVerts(gpuRefVerts);
    }

    // Reference Mesh Metrics
    {
        std::vector<GpuMetric> gpuRefMetrics;
        updateCudaRefMetrics(gpuRefMetrics);
    }
}

void LocalSampler::updateAnalyticalMetric(
        const Mesh& mesh)
{
    size_t vertCount = mesh.verts.size();

    std::vector<MeshMetric> metrics(vertCount);
    for(size_t vId=0; vId < vertCount; ++vId)
        metrics[vId] = vertMetric(mesh, vId);

    buildBackgroundMesh(mesh, metrics);
}

void LocalSampler::updateComputedMetric(
        const Mesh& mesh,
        const std::shared_ptr<LocalSampler>& sampler)
{

}

MeshMetric LocalSampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    const MeshLocalTet* tet = &_localTets[cachedRefTet];

    double coor[4];
    bool isUnreachable = false;

    while(!isUnreachable && !tetParams(_refVerts, *tet, position, coor))
    {
        glm::dvec3 vp[] = {
            _refVerts[tet->v[0]].p,
            _refVerts[tet->v[1]].p,
            _refVerts[tet->v[2]].p,
            _refVerts[tet->v[3]].p
        };

        glm::dvec3 orig = 0.25 * (vp[0] + vp[1] + vp[2] + vp[3]);
        glm::dvec3 dir = glm::normalize(orig - position);

        int t = 4;
        int trialCount = -1;
        while(t == 4 && !isUnreachable)
        {
            // Find exit face
            for(t = 0; t < 4; ++t)
            {
                if(triIntersect(
                    vp[MeshTet::tris[t].v[0]],
                    vp[MeshTet::tris[t].v[1]],
                    vp[MeshTet::tris[t].v[2]],
                    orig, dir))
                {
                    if(tet->n[t] != -1)
                        tet = &_localTets[tet->n[t]];
                    else
                        isUnreachable = true;

                    break;
                }
            }

            // If exit face not found
            if(t == 4)
            {
                // Start from an other position in the tet
                ++trialCount;

                // If there are still untried positions
                if(trialCount < 4)
                {
                    const double INV_MASS = 1.0 / 10.0;
                    const double WEIGHTS[] = {1.0, 2.0, 3.0, 4.0};

                    // Initialize ray from next position
                    orig = INV_MASS * (
                        WEIGHTS[(trialCount + 0) % 4] * vp[0] +
                        WEIGHTS[(trialCount + 1) % 4] * vp[1] +
                        WEIGHTS[(trialCount + 2) % 4] * vp[2] +
                        WEIGHTS[(trialCount + 3) % 4] * vp[3]);

                    dir = glm::normalize(orig - position);
                }
                else
                {
                    // Get projection on current tet
                    getLog().postMessage(new Message('E', false,
                        "Did not find a way out of the tet...",
                        "LocalSampler"));

                    isUnreachable = true;
                    break;
                }
            }
        }
    }


    if(isUnreachable)
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
    }

    return coor[0] * _refMetrics[tet->v[0]] +
           coor[1] * _refMetrics[tet->v[1]] +
           coor[2] * _refMetrics[tet->v[2]] +
           coor[3] * _refMetrics[tet->v[3]];
}

void LocalSampler::releaseDebugMesh()
{
    _debugMesh.reset();
}

const Mesh& LocalSampler::debugMesh()
{
    if(_debugMesh.get() == nullptr)
    {
        _debugMesh.reset(new Mesh());

        if(!_surfTris.empty())
        {
            for(const Triangle& tri : _surfTris)
            {
                size_t vertBase = _debugMesh->verts.size();
                _debugMesh->verts.push_back(_refVerts[tri.v[0]]);
                _debugMesh->verts.push_back(_refVerts[tri.v[1]]);
                _debugMesh->verts.push_back(_refVerts[tri.v[2]]);

                _debugMesh->tets.push_back(MeshTet(vertBase, vertBase+1, vertBase+2, vertBase));
            }

            for(const glm::dvec4& s : _failedSamples)
            {
                uint v = _debugMesh->verts.size();
                _debugMesh->verts.push_back(glm::dvec3(s));
                _debugMesh->tets.push_back(MeshTet(v, v, v, v));
                _debugMesh->tets.back().value = s.w;
            }

            _debugMesh->modelName = "Local Sampling Mesh";
            _debugMesh->compileTopology();
        }
    }

    return *_debugMesh;
}

void LocalSampler::buildBackgroundMesh(
        const Mesh& mesh,
        const std::vector<MeshMetric>& metrics)
{
    assert(metrics.size() == mesh.verts.size());

    // Clear resources
    _localTets.clear();
    _localTets.shrink_to_fit();
    _refVerts = mesh.verts;
    _refVerts.shrink_to_fit();
    _refMetrics = metrics;
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
        MeshLocalTet& tet = _localTets[t];
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

        mesh.verts[tet.v[0]].c = t;
        mesh.verts[tet.v[1]].c = t;
        mesh.verts[tet.v[2]].c = t;
        mesh.verts[tet.v[3]].c = t;
    }

    _surfTris = triSet.gather();
    size_t remTriCount = _surfTris.size();
    getLog().postMessage(new Message('I', false,
        "Surface triangle count: " + std::to_string(remTriCount) +
        " / " + std::to_string(triCount), "LocalSampler"));
    triSet.releaseMemoryPool();

    _failedSamples.clear();
    if(_debugMesh.get() != nullptr)
    {
        releaseDebugMesh();
        debugMesh();
    }

    _maxSearchDepth = 0;
}
