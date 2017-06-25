#include "TextureSampler.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <QImage>
#include <QPainter>
#include <QLabel>

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

const MeshMetric ISOTROPIC_METRIC(1.0);


class TextureGrid
{
public:
    TextureGrid(const glm::ivec3& size,
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

    inline MeshMetric& at(const glm::ivec3& pos)
    {
        return _impl[pos];
    }

    inline MeshMetric& at(int i, int j, int k)
    {
        return _impl.get(i, j, k);
    }

    const glm::ivec3 size;
    const glm::dvec3 extents;
    const glm::dvec3 minBounds;

    const glm::ivec3 minCellId;
    const glm::ivec3 maxCellId;

private:
    Grid3D<MeshMetric> _impl;
};


// CUDA Drivers Interface
void installCudaTextureSampler();
void updateCudaSamplerTextures(
        const std::vector<glm::vec4>& topLineBuff,
        const std::vector<glm::vec4>& sideTriBuff,
        const glm::mat4& texTransform,
        const glm::ivec3 texDims);


TextureSampler::TextureSampler(const std::string& name) :
    AbstractSampler(name, ":/glsl/compute/Sampling/Texture.glsl", installCudaTextureSampler),
    _topLineTex(0),
    _sideTriTex(0)
{
}

TextureSampler::TextureSampler() :
    AbstractSampler("Texture", ":/glsl/compute/Sampling/Texture.glsl", installCudaTextureSampler),
    _topLineTex(0),
    _sideTriTex(0)
{
}

TextureSampler::~TextureSampler()
{
    glDeleteTextures(1, &_topLineTex);
    _topLineTex = 0;
    glDeleteTextures(1, &_sideTriTex);
    _sideTriTex = 0;
}

bool TextureSampler::isMetricWise() const
{
    return true;
}

bool TextureSampler::useComputedMetric() const
{
    return false;
}

void TextureSampler::setPluginGlslUniforms(
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

void TextureSampler::setPluginCudaUniforms(
        const Mesh& mesh) const
{
    AbstractSampler::setPluginCudaUniforms(mesh);
}

void TextureSampler::updateGlslData(const Mesh& mesh) const
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
                const MeshMetric& metric = _grid->at(cellId);

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

void TextureSampler::updateCudaData(const Mesh& mesh) const
{
    glm::ivec3 size = _grid->size;
    size_t cellCount = size.x * size.y * size.z;

    std::vector<glm::vec4> topLineBuff;
    std::vector<glm::vec4> sideTriBuff;

    topLineBuff.reserve(cellCount);
    sideTriBuff.reserve(cellCount);

    for(int k = 0; k < size.z; ++k)
    {
        for(int j = 0; j < size.y; ++j)
        {
            for(int i = 0; i < size.x; ++i)
            {
                glm::ivec3 cellId(i, j, k);
                const MeshMetric& metric = _grid->at(cellId);

                glm::vec3 topline = metric[0];
                topLineBuff.push_back(glm::vec4(topline, 0));

                glm::vec3 sideTri(metric[1][1],
                                  metric[1][2],
                                  metric[2][2]);
                sideTriBuff.push_back(glm::vec4(sideTri, 0));
            }
        }
    }

    updateCudaSamplerTextures(topLineBuff, sideTriBuff, _transform, size);
}

void TextureSampler::clearGlslMemory(const Mesh& mesh) const
{
    glBindTexture(GL_TEXTURE_3D, _topLineTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F,
                 0, 0, 0, 0, GL_RGB, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_3D, _sideTriTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB32F,
                 0, 0, 0, 0,GL_RGB, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_3D, 0);
}

void TextureSampler::clearCudaMemory(const Mesh& mesh) const
{
    glm::ivec3 size = glm::ivec3(0, 0, 0);
    std::vector<glm::vec4> topLineBuff;
    std::vector<glm::vec4> sideTriBuff;

    updateCudaSamplerTextures(
        topLineBuff, sideTriBuff, _transform, size);
}

void TextureSampler::updateAnalyticalMetric(
        const Mesh& mesh)
{
    LocalSampler localSampler;
    localSampler.setScaling(scaling());
    localSampler.setAspectRatio(aspectRatio());

    localSampler.updateAnalyticalMetric(mesh);

    buildGrid(mesh, localSampler);
}

void TextureSampler::updateComputedMetric(
        const Mesh& mesh,
        const std::shared_ptr<LocalSampler>& sampler)
{
}

void TextureSampler::buildGrid(
        const Mesh& mesh,
        LocalSampler& sampler)
{
    _debugMesh.reset();

    if(mesh.verts.empty())
    {
        _grid.reset(
            new TextureGrid(
                glm::ivec3(1),
                glm::dvec3(2),
                glm::ivec3(-1)));
        return;
    }


    // Find grid bounds
    glm::dvec3 minBounds, maxBounds;
    boundingBox(mesh, minBounds, maxBounds);
    glm::dvec3 extents = maxBounds - minBounds;

    // Compute grid size
    int depth = discretizationDepth();
    size_t cellCount = mesh.verts.size();
    if(depth > 0)
        cellCount = depth * depth * depth;

    double alpha = glm::pow(cellCount / (extents.x*extents.y*extents.z), 1/3.0);
    glm::ivec3 size = glm::round(glm::max(glm::dvec3(1), alpha * extents));

    _transform = glm::scale(glm::mat4(),
        glm::vec3(1 / extents.x, 1 / extents.y, 1 / extents.z));
    _transform *= glm::translate(glm::mat4(),
        glm::vec3(-minBounds));

    _grid.reset(new TextureGrid(
        size, extents, minBounds));


    getLog().postMessage(new Message('I', false,
        "Sampling mesh metric in a texture",
        "TextureSampler"));
    getLog().postMessage(new Message('I', false,
        "Grid size: (" + std::to_string(size.x) + ", " +
                         std::to_string(size.y) + ", " +
                         std::to_string(size.z) + ")",
        "TextureSampler"));


    const auto& localTets = sampler.localTets();

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

                    _grid->at(id) = sampler.metricAt(pos, ev.cacheTetId);

                    cellIsSet.set(i, j, k, true);
                }
            }
        }
    }

    /*
    int zoom = 4;
    glm::dvec2 zoom2 = glm::dvec2(size*zoom);
    int halfZ = size.z / 2;
    QImage image(size.x*zoom, size.y*zoom, QImage::Format_RGB32);
    QPainter painter(&image);
    QBrush brush(QColor(70, 70, 70));

    painter.fillRect(0, 0, size.x*zoom, size.y*zoom, brush);

    // Metric
    for(int j=0; j< size.y; ++j)
    {
        for(int i=0; i < size.x; ++i)
        {
            double m = _grid->at(i,j,halfZ)[0][0];
            if(m != 1.0)
            {
                brush.setColor(QColor((26.0 / sqrt(m))*255, 0, 0));
                painter.fillRect(i*zoom, j*zoom, zoom, zoom, brush);
            }
        }
    }

    // Grid
    painter.setPen(Qt::black);
    for(int j=0; j< size.y; ++j)
    {
        for(int i=0; i < size.x; ++i)
        {
            painter.drawLine(i*zoom, 0, i*zoom, size.y*zoom);
            painter.drawLine(0, j*zoom, size.x*zoom, j*zoom);
        }
    }

    // Geometry
    std::vector<glm::dvec2> geo;
    geo.push_back(glm::dvec2(-1,0.2));
    geo.push_back(glm::dvec2(-1,0.8));
    geo.push_back(glm::dvec2(0.5,0.8));
    for(double a=0; a < glm::pi<double>(); a+=glm::pi<double>()/100.0)
        geo.push_back(glm::dvec2(0.5 + 0.8*sin(a), 0.8*cos(a)));
    geo.push_back(glm::dvec2(0.5,-0.8));
    geo.push_back(glm::dvec2(-1,-0.8));
    geo.push_back(glm::dvec2(-1,-0.2));
    geo.push_back(glm::dvec2(0.5,-0.2));
    for(double a=0; a < glm::pi<double>(); a+=glm::pi<double>()/100.0)
        geo.push_back(glm::dvec2(0.5 + 0.2*sin(a), -0.2*cos(a)));
    geo.push_back(glm::dvec2(0.5,0.2));
    geo.push_back(glm::dvec2(-1,0.2));

    QPen pen(Qt::cyan);
    pen.setWidthF(2.0);
    painter.setPen(pen);
    for(int i=0; i < geo.size(); ++i)
    {
        int ip = (i+1) % geo.size();
        glm::dvec2 x1 = glm::dvec2(_transform * glm::dvec4(geo[i], 0, 1)) * zoom2;
        glm::dvec2 x2 = glm::dvec2(_transform * glm::dvec4(geo[ip], 0, 1)) * zoom2;
        painter.drawLine(x1.x, x1.y, x2.x, x2.y);
    }

    QLabel* label = new QLabel();
    label->setPixmap(QPixmap::fromImage(image));
    label->show();
    */
}

MeshMetric TextureSampler::metricAt(
        const glm::dvec3& position,
        uint& cachedRefTet) const
{
    glm::dvec3 cs = _grid->extents / glm::dvec3(_grid->size);
    glm::dvec3 minB = _grid->minBounds;
    glm::dvec3 maxB = _grid->minBounds + _grid->extents - (cs *1.5);
    glm::dvec3 cp0 = glm::clamp(position + glm::dvec3(-cs.x, -cs.y, -cs.z)/2.0, minB, maxB);

    glm::ivec3 id0 = cellId(*_grid, cp0);
    glm::ivec3 id1 = glm::min(
        id0 + glm::ivec3(1, 1, 1),
        _grid->size - glm::ivec3(1, 1, 1));

    MeshMetric m000 = _grid->at(id0.x, id0.y, id0.z);
    MeshMetric m100 = _grid->at(id1.x, id0.y, id0.z);
    MeshMetric m010 = _grid->at(id0.x, id1.y, id0.z);
    MeshMetric m110 = _grid->at(id1.x, id1.y, id0.z);
    MeshMetric m001 = _grid->at(id0.x, id0.y, id1.z);
    MeshMetric m101 = _grid->at(id1.x, id0.y, id1.z);
    MeshMetric m011 = _grid->at(id0.x, id1.y, id1.z);
    MeshMetric m111 = _grid->at(id1.x, id1.y, id1.z);

    glm::dvec3 c0Center = cs * (glm::dvec3(id0) + glm::dvec3(0.5));
    glm::dvec3 a = (position - (_grid->minBounds + c0Center)) / cs;
    a = glm::clamp(a, glm::dvec3(0), glm::dvec3(1));

    MeshMetric mx00 = interpolateMetrics(m000, m100, a.x);
    MeshMetric mx10 = interpolateMetrics(m010, m110, a.x);
    MeshMetric mx01 = interpolateMetrics(m001, m101, a.x);
    MeshMetric mx11 = interpolateMetrics(m011, m111, a.x);

    MeshMetric mxy0 = interpolateMetrics(mx00, mx10, a.y);
    MeshMetric mxy1 = interpolateMetrics(mx01, mx11, a.y);

    MeshMetric mxyz = interpolateMetrics(mxy0, mxy1, a.z);

    return mxyz;
}

void TextureSampler::releaseDebugMesh()
{
    _debugMesh.reset();
}

const Mesh& TextureSampler::debugMesh()
{
    if(_debugMesh.get() == nullptr)
    {
        _debugMesh.reset(new Mesh());

        if(_grid.get() != nullptr)
        {
            meshGrid(*_grid.get(), *_debugMesh);

            _debugMesh->modelName = "Texture Sampling Mesh";
            _debugMesh->compileTopology();
        }
    }

    return *_debugMesh;
}

inline glm::ivec3 TextureSampler::cellId(
        const TextureGrid& grid,
        const glm::dvec3& vertPos) const
{
    glm::dvec3 origDist = vertPos - grid.minBounds;
    glm::dvec3 distRatio = origDist / grid.extents;

    glm::ivec3 cellId = glm::ivec3(distRatio * glm::dvec3(grid.size));
    cellId = glm::clamp(cellId, _grid->minCellId, _grid->maxCellId);

    return cellId;
}

void TextureSampler::meshGrid(TextureGrid& grid, Mesh& mesh)
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

                    hex.value = 26.0 / sqrt(_grid->at(cellId)[0][0]);
                    mesh.hexs.push_back(hex);
                }
            }
        }
    }
}
