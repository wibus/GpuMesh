#include "UniformDiscretizer.h"

#include <CellarWorkbench/DataStructure/Grid3D.h>
#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/Mesh.h"

using namespace cellar;



struct ElemValue
{
    ElemValue() :
        minBox(INT32_MAX),
        maxBox(INT32_MIN),
        metric(0.0)
    {
    }

    glm::ivec3 minBox;
    glm::ivec3 maxBox;
    Metric metric;
};

const Metric ISOTROPIC_METRIC(1.0);

struct GridCell
{
    GridCell() : metric(0), weight(0) {}

    Metric metric;
    double weight;
};

class UniformGrid
{
public:
    UniformGrid(const glm::ivec3& size,
                const glm::dvec3& extents,
                const glm::dvec3& minBounds):
        size(size),
        extents(extents),
        minBounds(minBounds),
        _impl(size.x, size.y, size.z)
    {
    }

    inline GridCell& at(const glm::ivec3& pos)
    {
        return _impl[pos];
    }

    const glm::ivec3 size;
    const glm::dvec3 extents;
    const glm::dvec3 minBounds;

private:
    Grid3D<GridCell> _impl;
};


UniformDiscretizer::UniformDiscretizer() :
    AbstractDiscretizer("Uniform", "")
{
}

UniformDiscretizer::~UniformDiscretizer()
{

}

bool UniformDiscretizer::isMetricWise() const
{
    return true;
}

void UniformDiscretizer::installPlugIn(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractDiscretizer::installPlugIn(mesh, program);
}

void UniformDiscretizer::uploadUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    AbstractDiscretizer::uploadUniforms(mesh, program);
}

void UniformDiscretizer::discretize(const Mesh& mesh, int density)
{
    _debugMesh.reset();
    if(mesh.verts.empty())
    {
        _grid.reset();
    }


    // Find grid bounds
    glm::dvec3 minBounds, maxBounds;
    boundingBox(mesh, minBounds, maxBounds);
    glm::dvec3 extents = maxBounds - minBounds;

    // Compute grid size
    size_t vertCount = mesh.verts.size();
    double alpha = glm::pow(vertCount / (density * extents.x*extents.y*extents.z), 1/3.0);
    glm::ivec3 gridSize(alpha * extents);

    getLog().postMessage(new Message('I', false,
        "Discretizing mesh metric in a Uniform grid",
        "UniformDiscretizer"));
    getLog().postMessage(new Message('I', false,
        "Grid size: (" + std::to_string(gridSize.x) + ", " +
                         std::to_string(gridSize.y) + ", " +
                         std::to_string(gridSize.z) + ")",
        "UniformDiscretizer"));


    std::vector<ElemValue> elemValues;

    size_t tetCount = mesh.tets.size();
    for(size_t e=0; e < tetCount; ++e)
    {
        ElemValue ev;
        const MeshTet& elem = mesh.tets[e];
        glm::dvec3 minBoxPos = glm::dvec3(INFINITY);
        glm::dvec3 maxBoxPos = glm::dvec3(-INFINITY);
        for(uint v=0; v < MeshTet::VERTEX_COUNT; ++v)
        {
            uint vId = elem.v[v];
            const glm::dvec3& vertPos = mesh.verts[vId].p;
            minBoxPos = glm::min(minBoxPos, vertPos);
            maxBoxPos = glm::max(maxBoxPos, vertPos);
            ev.metric += vertMetric(mesh, vId);
        }

        ev.minBox = cellId(gridSize, minBounds, extents, minBoxPos);
        ev.maxBox = cellId(gridSize, minBounds, extents, maxBoxPos);
        ev.metric /= MeshTet::VERTEX_COUNT;
        elemValues.push_back(ev);
    }

    size_t priCount = mesh.pris.size();
    for(size_t e=0; e < priCount; ++e)
    {
        ElemValue ev;
        const MeshPri& elem = mesh.pris[e];
        glm::dvec3 minBoxPos = glm::dvec3(INFINITY);
        glm::dvec3 maxBoxPos = glm::dvec3(-INFINITY);
        for(uint v=0; v < MeshPri::VERTEX_COUNT; ++v)
        {
            uint vId = elem.v[v];
            const glm::dvec3& vertPos = mesh.verts[vId].p;
            minBoxPos = glm::min(minBoxPos, vertPos);
            maxBoxPos = glm::max(maxBoxPos, vertPos);
            ev.metric += vertMetric(mesh, vId);
        }

        ev.minBox = cellId(gridSize, minBounds, extents, minBoxPos);
        ev.maxBox = cellId(gridSize, minBounds, extents, maxBoxPos);
        ev.metric /= MeshPri::VERTEX_COUNT;
        elemValues.push_back(ev);
    }

    size_t hexCount = mesh.hexs.size();
    for(size_t e=0; e < hexCount; ++e)
    {
        ElemValue ev;
        const MeshHex& elem = mesh.hexs[e];
        glm::dvec3 minBoxPos = glm::dvec3(INFINITY);
        glm::dvec3 maxBoxPos = glm::dvec3(-INFINITY);
        for(uint v=0; v < MeshHex::VERTEX_COUNT; ++v)
        {
            uint vId = elem.v[v];
            const glm::dvec3& vertPos = mesh.verts[vId].p;
            minBoxPos = glm::min(minBoxPos, vertPos);
            maxBoxPos = glm::max(maxBoxPos, vertPos);
            ev.metric += vertMetric(mesh, vId);
        }

        ev.minBox = cellId(gridSize, minBounds, extents, minBoxPos);
        ev.maxBox = cellId(gridSize, minBounds, extents, maxBoxPos);
        ev.metric /= MeshHex::VERTEX_COUNT;
        elemValues.push_back(ev);
    }


    _grid.reset(new UniformGrid(gridSize, extents, minBounds));

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
                    glm::ivec3 id(i, j, k);
                    GridCell& cell = _grid->at(id);
                    cell.metric += ev.metric;
                    cell.weight += 1.0;
                }
            }
        }
    }

    // Divide each cell's value by its weight
    for(int k=0; k < gridSize.z; ++k)
    {
        for(int j=0; j < gridSize.y; ++j)
        {
            for(int i=0; i < gridSize.x; ++i)
            {
                glm::ivec3 id(i, j, k);
                GridCell& cell = _grid->at(id);
                if(cell.weight > 0.0)
                {
                    cell.metric /= cell.weight;
                }
                else
                {
                    cell.metric = ISOTROPIC_METRIC;
                }
            }
        }
    }
}

double UniformDiscretizer::distance(
        const glm::dvec3& a,
        const glm::dvec3& b) const
{
    glm::dvec3 d = a - b;
    glm::dvec3 m = (a + b) / 2.0;
    return glm::sqrt(glm::dot(d, metric(m) * d));
}

void UniformDiscretizer::releaseDebugMesh()
{
    _debugMesh.reset();
}

const Mesh& UniformDiscretizer::debugMesh()
{
    if(_debugMesh.get() == nullptr)
    {
        _debugMesh.reset(new Mesh());

        if(_grid.get() != nullptr)
        {
            meshGrid(*_grid.get(), *_debugMesh);

            _debugMesh->modelName = "Uniform Discretization Mesh";
            _debugMesh->compileTopology();
        }
    }

    return *_debugMesh;
}

Metric UniformDiscretizer::metric(
        const glm::dvec3& position) const
{
    glm::dvec3 cs = _grid->extents / glm::dvec3(_grid->size);
    glm::dvec3 minB = _grid->minBounds;
    glm::dvec3 maxB = _grid->minBounds + _grid->extents - (cs *1.5);
    glm::dvec3 cp000 = glm::clamp(position + glm::dvec3(-cs.x, -cs.y, -cs.z)/2.0, minB, maxB);

    glm::ivec3 id000 = cellId(_grid->size, _grid->minBounds, _grid->extents, cp000);

    glm::ivec3 id100 = id000 + glm::ivec3(1, 0, 0);
    glm::ivec3 id010 = id000 + glm::ivec3(0, 1, 0);
    glm::ivec3 id110 = id000 + glm::ivec3(1, 1, 0);
    glm::ivec3 id001 = id000 + glm::ivec3(0, 0, 1);
    glm::ivec3 id101 = id000 + glm::ivec3(1, 0, 1);
    glm::ivec3 id011 = id000 + glm::ivec3(0, 1, 1);
    glm::ivec3 id111 = id000 + glm::ivec3(1, 1, 1);

    Metric m000 = _grid->at(id000).metric;
    Metric m100 = _grid->at(id100).metric;
    Metric m010 = _grid->at(id010).metric;
    Metric m110 = _grid->at(id110).metric;
    Metric m001 = _grid->at(id001).metric;
    Metric m101 = _grid->at(id101).metric;
    Metric m011 = _grid->at(id011).metric;
    Metric m111 = _grid->at(id111).metric;

    glm::dvec3 c000Center = cs * (glm::dvec3(id000) + glm::dvec3(0.5));
    glm::dvec3 a = (position - (_grid->minBounds + c000Center)) / cs;

    Metric mx00 = interpolate(m000, m100, a.x);
    Metric mx10 = interpolate(m010, m110, a.x);
    Metric mx01 = interpolate(m001, m101, a.x);
    Metric mx11 = interpolate(m011, m111, a.x);

    Metric mxy0 = interpolate(mx00, mx10, a.y);
    Metric mxy1 = interpolate(mx01, mx11, a.y);

    Metric mxyz = interpolate(mxy0, mxy1, a.z);

    return mxyz;
}

inline glm::ivec3 UniformDiscretizer::cellId(
        const glm::ivec3& gridSize,
        const glm::dvec3& minBounds,
        const glm::dvec3& extents,
        const glm::dvec3& vertPos) const
{
    glm::dvec3 origDist = vertPos - minBounds;
    glm::dvec3 distRatio = origDist / extents;
    glm::ivec3 cellId = glm::ivec3(distRatio * glm::dvec3(gridSize));
    cellId = glm::clamp(cellId, glm::ivec3(), gridSize - glm::ivec3(1));
    return cellId;
}

void UniformDiscretizer::meshGrid(UniformGrid& grid, Mesh& mesh)
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
                   grid.at(cellId).metric != ISOTROPIC_METRIC)
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
                        xb + yt + zb,
                        xt + yt + zb,
                        xb + yb + zt,
                        xt + yb + zt,
                        xb + yt + zt,
                        xt + yt + zt);

                    hex.value = grid.at(cellId).metric[0][0];
                    mesh.hexs.push_back(hex);
                }
            }
        }
    }
}
