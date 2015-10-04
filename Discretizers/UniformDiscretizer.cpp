#include "UniformDiscretizer.h"

#include <CellarWorkbench/DataStructure/Grid3D.h>
#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/Mesh.h"

using namespace cellar;


struct GridCell
{
    GridCell() :
        value(0.0),
        weight(0.0)
    {
    }

    double value;
    double weight;
};

struct ElemValue
{
    ElemValue() :
        minBox(INT32_MAX),
        maxBox(INT32_MIN),
        value()
    {
    }

    glm::ivec3 minBox;
    glm::ivec3 maxBox;
    double value;
};

UniformDiscretizer::UniformDiscretizer() :
    _gridMesh(new Mesh())
{
    _gridMesh->modelName = "Uniform Discretization Grid";
}

UniformDiscretizer::~UniformDiscretizer()
{

}

std::shared_ptr<Mesh> UniformDiscretizer::gridMesh() const
{
    return _gridMesh;
}

void UniformDiscretizer::discretize(
        const Mesh& mesh,
        const glm::ivec3& gridSize)
{
    getLog().postMessage(new Message('I', false,
        "Discretizing mesh metric in a Uniform grid",
        "UniformDiscretizer"));
    getLog().postMessage(new Message('I', false,
        "Grid size: (" + std::to_string(gridSize.x) + ", " +
                         std::to_string(gridSize.y) + ", " +
                         std::to_string(gridSize.z) + ")",
        "UniformDiscretizer"));

    // Find grid bounds
    glm::dvec3 minBounds, maxBounds;
    boundingBox(mesh, minBounds, maxBounds);
    glm::dvec3 extents = maxBounds - minBounds;


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
            ev.value += vertValue(mesh, vId);
        }

        ev.minBox = cellId(gridSize, minBounds, extents, minBoxPos);
        ev.maxBox = cellId(gridSize, minBounds, extents, maxBoxPos);
        ev.value /= MeshTet::VERTEX_COUNT;
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
            ev.value += vertValue(mesh, vId);
        }

        ev.minBox = cellId(gridSize, minBounds, extents, minBoxPos);
        ev.maxBox = cellId(gridSize, minBounds, extents, maxBoxPos);
        ev.value /= MeshPri::VERTEX_COUNT;
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
            ev.value += vertValue(mesh, vId);
        }

        ev.minBox = cellId(gridSize, minBounds, extents, minBoxPos);
        ev.maxBox = cellId(gridSize, minBounds, extents, maxBoxPos);
        ev.value /= MeshHex::VERTEX_COUNT;
        elemValues.push_back(ev);
    }


    Grid3D<GridCell> weightedMeanGrid(gridSize.x, gridSize.y, gridSize.z);

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
                    GridCell& cell = weightedMeanGrid[id];
                    cell.value += ev.value;
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
                GridCell& cell = weightedMeanGrid[id];
                if(cell.weight > 0.0)
                {
                    cell.value /= cell.weight;
                }
                else
                {
                    cell.value = -1.0;
                }
            }
        }
    }

    // Build the grid mesh for visualization
    _gridMesh->clear();
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

                _gridMesh->verts.push_back(MeshVert(
                    glm::dvec3(minBounds + cellPos * extents)));

                if(glm::all(glm::lessThan(cellId, gridSize)) &&
                   weightedMeanGrid[cellId].value >= 0.0)
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

                    hex.value = weightedMeanGrid[cellId].value;
                    _gridMesh->hexs.push_back(hex);
                }
            }
        }
    }

    _gridMesh->compileTopology();
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
