#include "UniformSampler.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/DataStructure/Grid3D.h>
#include <CellarWorkbench/GL/GlProgram.h>
#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/Mesh.h"

#include "LocalSampler.h"

using namespace cellar;



struct ElemValue
{
    ElemValue() :
        minBox(INT32_MAX),
        maxBox(INT32_MIN),
        cacheTetId(-1)
    {
    }

    glm::ivec3 minBox;
    glm::ivec3 maxBox;
    uint cacheTetId;
};

const Metric ISOTROPIC_METRIC(1.0);


class UniformGrid
{
public:
    UniformGrid(const glm::ivec3& size,
                const glm::dvec3& extents,
                const glm::dvec3& minBounds):
        size(size),
        extents(extents),
        minBounds(minBounds),
        minCellId(0, 0, 0),
        maxCellId(size - glm::ivec3(1)),
        _impl(size.x, size.y, size.z)
    {
    }

    inline Metric& at(const glm::ivec3& pos)
    {
        return _impl[pos];
    }

    const glm::ivec3 size;
    const glm::dvec3 extents;
    const glm::dvec3 minBounds;

    const glm::ivec3 minCellId;
    const glm::ivec3 maxCellId;

private:
    Grid3D<Metric> _impl;
};


// CUDA Drivers Interface
void installCudaUniformSampler();


UniformSampler::UniformSampler() :
    AbstractSampler("Uniform", ":/glsl/compute/Sampling/Uniform.glsl", installCudaUniformSampler),
    _topLineTex(0),
    _sideTriTex(0)
{
}

UniformSampler::~UniformSampler()
{
    glDeleteTextures(1, &_topLineTex);
    _topLineTex = 0;
    glDeleteTextures(1, &_sideTriTex);
    _sideTriTex = 0;
}

bool UniformSampler::isMetricWise() const
{
    return true;
}

void UniformSampler::setPluginGlslUniforms(
        const Mesh& mesh,
        const cellar::GlProgram& program) const
{
    AbstractSampler::setPluginGlslUniforms(mesh, program);

    program.pushProgram();
    program.setInt("TopLineTex", 0);
    program.setInt("SideTriTex", 1);
    program.setMat4f("TexTransform", _transform);
    program.popProgram();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, _sideTriTex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, _topLineTex);
}

void UniformSampler::setPluginCudaUniforms(
        const Mesh& mesh) const
{
    AbstractSampler::setPluginCudaUniforms(mesh);
}

void UniformSampler::updateGlslData(const Mesh& mesh) const
{
    if(_topLineTex == 0)
        glGenTextures(1, &_topLineTex);

    if(_sideTriTex == 0)
        glGenTextures(1, &_sideTriTex);


    glm::ivec3 size = _grid->size;
    size_t cellCount = size.x * size.y * size.z;

    std::vector<glm::vec3> topLineBuff;
    std::vector<glm::vec3> sideTriBuff;

    topLineBuff.reserve(cellCount);
    sideTriBuff.reserve(cellCount);

    for(int k = 0; k < size.z; ++k)
    {
        for(int j = 0; j < size.y; ++j)
        {
            for(int i = 0; i < size.x; ++i)
            {
                glm::ivec3 cellId(i, j, k);
                const Metric& metric = _grid->at(cellId);

                glm::vec3 topline = metric[0];
                topLineBuff.push_back(topline);

                glm::vec3 sideTri(metric[1][1],
                                  metric[1][2],
                                  metric[2][2]);
                sideTriBuff.push_back(sideTri);
            }
        }
    }

    glBindTexture(GL_TEXTURE_3D, _topLineTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F,
                 size.x, size.y, size.z, 0,
                 GL_RGB, GL_FLOAT, topLineBuff.data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_3D, _sideTriTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F,
                 size.x, size.y, size.z, 0,
                 GL_RGB, GL_FLOAT, sideTriBuff.data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_3D, 0);
}

void UniformSampler::updateCudaData(const Mesh& mesh) const
{

}

void UniformSampler::clearGlslMemory(const Mesh& mesh) const
{

}

void UniformSampler::clearCudaMemory(const Mesh& mesh) const
{

}

void UniformSampler::setReferenceMesh(
        const Mesh& mesh)
{
    _debugMesh.reset();


    // Find grid bounds
    glm::dvec3 minBounds, maxBounds;
    boundingBox(mesh, minBounds, maxBounds);
    glm::dvec3 extents = maxBounds - minBounds;

    // Compute grid size
    size_t vertCount = mesh.verts.size();
    double alpha = glm::pow(vertCount / (2 * extents.x*extents.y*extents.z), 1/3.0);
    glm::ivec3 size(alpha * extents);

    _transform = glm::scale(glm::mat4(),
        glm::vec3(1 / extents.x, 1 / extents.y, 1 / extents.z));
    _transform *= glm::translate(glm::mat4(),
        glm::vec3(-minBounds));

    _grid.reset(new UniformGrid(
        size, extents, minBounds));


    getLog().postMessage(new Message('I', false,
        "Sampling mesh metric in a Uniform grid",
        "UniformSampler"));
    getLog().postMessage(new Message('I', false,
        "Grid size: (" + std::to_string(size.x) + ", " +
                         std::to_string(size.y) + ", " +
                         std::to_string(size.z) + ")",
        "UniformSampler"));

    LocalSampler localSampler;
    localSampler.setScaling(scaling());
    localSampler.setReferenceMesh(mesh);
    const auto& localTets = localSampler.localTets();

    std::vector<ElemValue> elemValues;
    size_t tetCount = localTets.size();
    for(size_t e=0; e < tetCount; ++e)
    {
        ElemValue ev;
        const MeshLocalTet& elem = localTets[e];
        glm::dvec3 minBoxPos = glm::dvec3(INFINITY);
        glm::dvec3 maxBoxPos = glm::dvec3(-INFINITY);
        for(uint v=0; v < MeshTet::VERTEX_COUNT; ++v)
        {
            uint vId = elem.v[v];
            const glm::dvec3& vertPos = mesh.verts[vId].p;
            minBoxPos = glm::min(minBoxPos, vertPos);
            maxBoxPos = glm::max(maxBoxPos, vertPos);
        }

        ev.cacheTetId = e;
        ev.minBox = cellId(*_grid, minBoxPos) - glm::ivec3(1);
        ev.maxBox = cellId(*_grid, maxBoxPos) + glm::ivec3(1);
        ev.minBox = glm::max(ev.minBox, _grid->minCellId);
        ev.maxBox = glm::min(ev.maxBox, _grid->maxCellId);
        elemValues.push_back(ev);
    }


    glm::dvec3 cellExtents = extents / glm::dvec3(size);
    Grid3D<bool> cellIsSet(size.x, size.y, size.z, false);

    size_t elemCount = elemValues.size();
    for(size_t e=0; e < elemCount; ++e)
    {
        ElemValue& ev = elemValues[e];

        for(int k=ev.minBox.z; k <= ev.maxBox.z; ++k)
        {
            for(int j=ev.minBox.y; j <= ev.maxBox.y; ++j)
            {
                for(int i=ev.minBox.x; i <= ev.maxBox.x; ++i)
                {
                    if(cellIsSet.get(i, j, k))
                        continue;

                    glm::ivec3 id(i, j, k);
                    glm::dvec3 pos = _grid->minBounds + cellExtents *
                        (glm::dvec3(id) + glm::dvec3(0.5));

                    _grid->at(id) = localSampler.metricAt(pos, ev.cacheTetId);
                    cellIsSet.set(i, j, k, true);
                }
            }
        }
    }
}

Metric UniformSampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    glm::dvec3 cs = _grid->extents / glm::dvec3(_grid->size);
    glm::dvec3 minB = _grid->minBounds;
    glm::dvec3 maxB = _grid->minBounds + _grid->extents - (cs *1.5);
    glm::dvec3 cp000 = glm::clamp(position + glm::dvec3(-cs.x, -cs.y, -cs.z)/2.0, minB, maxB);

    glm::ivec3 id000 = cellId(*_grid, cp000);

    glm::ivec3 id100 = id000 + glm::ivec3(1, 0, 0);
    glm::ivec3 id010 = id000 + glm::ivec3(0, 1, 0);
    glm::ivec3 id110 = id000 + glm::ivec3(1, 1, 0);
    glm::ivec3 id001 = id000 + glm::ivec3(0, 0, 1);
    glm::ivec3 id101 = id000 + glm::ivec3(1, 0, 1);
    glm::ivec3 id011 = id000 + glm::ivec3(0, 1, 1);
    glm::ivec3 id111 = id000 + glm::ivec3(1, 1, 1);

    Metric m000 = _grid->at(id000);
    Metric m100 = _grid->at(id100);
    Metric m010 = _grid->at(id010);
    Metric m110 = _grid->at(id110);
    Metric m001 = _grid->at(id001);
    Metric m101 = _grid->at(id101);
    Metric m011 = _grid->at(id011);
    Metric m111 = _grid->at(id111);

    glm::dvec3 c000Center = cs * (glm::dvec3(id000) + glm::dvec3(0.5));
    glm::dvec3 a = (position - (_grid->minBounds + c000Center)) / cs;
    a = glm::clamp(a, glm::dvec3(0), glm::dvec3(1));

    Metric mx00 = interpolateMetrics(m000, m100, a.x);
    Metric mx10 = interpolateMetrics(m010, m110, a.x);
    Metric mx01 = interpolateMetrics(m001, m101, a.x);
    Metric mx11 = interpolateMetrics(m011, m111, a.x);

    Metric mxy0 = interpolateMetrics(mx00, mx10, a.y);
    Metric mxy1 = interpolateMetrics(mx01, mx11, a.y);

    Metric mxyz = interpolateMetrics(mxy0, mxy1, a.z);

    return mxyz;
}

void UniformSampler::releaseDebugMesh()
{
    _debugMesh.reset();
}

const Mesh& UniformSampler::debugMesh()
{
    if(_debugMesh.get() == nullptr)
    {
        _debugMesh.reset(new Mesh());

        if(_grid.get() != nullptr)
        {
            meshGrid(*_grid.get(), *_debugMesh);

            _debugMesh->modelName = "Uniform Sampling Mesh";
            _debugMesh->compileTopology();
        }
    }

    return *_debugMesh;
}

inline glm::ivec3 UniformSampler::cellId(
        const UniformGrid& grid,
        const glm::dvec3& vertPos) const
{
    glm::dvec3 origDist = vertPos - grid.minBounds;
    glm::dvec3 distRatio = origDist / grid.extents;

    glm::ivec3 cellId = glm::ivec3(distRatio * glm::dvec3(grid.size));
    cellId = glm::clamp(cellId, _grid->minCellId, _grid->maxCellId);

    return cellId;
}

void UniformSampler::meshGrid(UniformGrid& grid, Mesh& mesh)
{
    const glm::ivec3 gridSize(grid.size);

    const int ROW_SIZE = gridSize.x + 1;
    const int LEVEL_SIZE = ROW_SIZE * (gridSize.y+1);

    for(int k=0; k <= gridSize.z; ++k)
    {
        for(int j=0; j <= gridSize.y; ++j)
        {
            for(int i=0; i <= gridSize.x; ++i)
            {
                glm::ivec3 cellId = glm::ivec3(i, j, k);
                glm::dvec3 cellPos = glm::dvec3(cellId) /
                                     glm::dvec3(gridSize);

                mesh.verts.push_back(MeshVert(
                    glm::dvec3(grid.minBounds + cellPos * grid.extents)));

                if(glm::all(glm::lessThan(cellId, gridSize)) &&
                   grid.at(cellId) != ISOTROPIC_METRIC)
                {
                    int xb = i;
                    int xt = i+1;
                    int yb = j*ROW_SIZE;
                    int yt = (j+1)*ROW_SIZE;
                    int zb = k*LEVEL_SIZE;
                    int zt = (k+1)*LEVEL_SIZE;
                    MeshHex hex(
                        xb + yb + zb,
                        xt + yb + zb,
                        xt + yt + zb,
                        xb + yt + zb,
                        xb + yb + zt,
                        xt + yb + zt,
                        xt + yt + zt,
                        xb + yt + zt);

                    hex.value = _grid->at(cellId)[0][0] /
                            (scaling() * scaling() * 10.0);
                    mesh.hexs.push_back(hex);
                }
            }
        }
    }
}
