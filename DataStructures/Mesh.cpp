#include "Mesh.h"

#include <algorithm>
#include <iostream>
#include <cstdint>

#include <CellarWorkbench/Misc/Log.h>

#include "Boundaries/BoundaryFree.h"

#include "Evaluators/AbstractEvaluator.h"

#include "OptimizationPlot.h"

#include "NodeGroups.h"

using namespace std;
using namespace cellar;


const MeshEdge MeshTet::edges[MeshTet::EDGE_COUNT] = {
    MeshEdge(0, 1),
    MeshEdge(0, 2),
    MeshEdge(0, 3),
    MeshEdge(1, 2),
    MeshEdge(1, 3),
    MeshEdge(2, 3)
};

const MeshTri MeshTet::tris[MeshTet::TRI_COUNT] = {
    MeshTri(1, 2, 3),
    MeshTri(0, 3, 2),
    MeshTri(0, 1, 3),
    MeshTri(0, 2, 1),
};

const MeshTet MeshTet::tets[MeshTet::TET_COUNT] = {
    MeshTet(0, 1, 2, 3)
};


const MeshEdge MeshPyr::edges[MeshPyr::EDGE_COUNT] = {
    MeshEdge(0, 1),
    MeshEdge(1, 2),
    MeshEdge(2, 3),
    MeshEdge(3, 0),
    MeshEdge(0, 4),
    MeshEdge(1, 4),
    MeshEdge(2, 4),
    MeshEdge(3, 4)
};

const MeshTri MeshPyr::tris[MeshPyr::TRI_COUNT] = {
    MeshTri(3, 1, 0), // Z neg face 0
    MeshTri(1, 3, 2), // Z neg face 1
    MeshTri(0, 1, 4), // Front
    MeshTri(1, 2, 4), // Right
    MeshTri(2, 3, 4), // Back
    MeshTri(3, 0, 4), // Left
};

const MeshTet MeshPyr::tets[MeshPyr::TET_COUNT] = {
    MeshTet(0, 1, 2, 4),
    MeshTet(0, 2, 3, 4),
};


const MeshEdge MeshPri::edges[MeshPri::EDGE_COUNT] = {
    MeshEdge(0, 1),
    MeshEdge(0, 2),
    MeshEdge(1, 2),
    MeshEdge(0, 3),
    MeshEdge(1, 4),
    MeshEdge(2, 5),
    MeshEdge(3, 4),
    MeshEdge(3, 5),
    MeshEdge(4, 5)
};

const MeshTri MeshPri::tris[MeshPri::TRI_COUNT] = {
    MeshTri(2, 1, 0), // Z neg face
    MeshTri(3, 4, 5), // Z pos face
    MeshTri(3, 2, 0), // Back face 0
    MeshTri(2, 3, 5), // Back face 1
    MeshTri(1, 3, 0), // Left face 0
    MeshTri(3, 1, 4), // Left face 1
    MeshTri(2, 4, 1), // Right face 0
    MeshTri(4, 2, 5)  // Right face 1
};

const MeshTet MeshPri::tets[MeshPri::TET_COUNT] = {
    MeshTet(0, 1, 2, 3),
    MeshTet(3, 5, 4, 2),
    MeshTet(1, 4, 2, 3)
};


const MeshEdge MeshHex::edges[MeshHex::EDGE_COUNT] = {
    MeshEdge(0, 1),
    MeshEdge(0, 3),
    MeshEdge(1, 2),
    MeshEdge(2, 3),
    MeshEdge(0, 4),
    MeshEdge(1, 5),
    MeshEdge(2, 6),
    MeshEdge(3, 7),
    MeshEdge(4, 5),
    MeshEdge(4, 7),
    MeshEdge(5, 6),
    MeshEdge(6, 7)
};

const MeshTri MeshHex::tris[MeshHex::TRI_COUNT] = {
    MeshTri(3, 1, 0), // Z neg face 0
    MeshTri(1, 3, 2), // Z neg face 1
    MeshTri(5, 7, 4), // Z pos face 0
    MeshTri(7, 5, 6), // Z pos face 1
    MeshTri(1, 4, 0), // Y neg face 0
    MeshTri(4, 1, 5), // Y neg face 1
    MeshTri(3, 6, 2), // Y pos face 0
    MeshTri(6, 3, 7), // Y pos face 1
    MeshTri(4, 3, 0), // X neg face 0
    MeshTri(3, 4, 7), // X neg face 1
    MeshTri(2, 5, 1), // X pos face 0
    MeshTri(5, 2, 6), // X pos face 1
};

const MeshTet MeshHex::tets[MeshHex::TET_COUNT] = {
    MeshTet(1, 4, 3, 0),
    MeshTet(1, 3, 6, 2),
    MeshTet(1, 6, 4, 5),
    MeshTet(3, 4, 6, 7),
    MeshTet(1, 3, 4, 6)
};


const AbstractConstraint* MeshTopo::NO_BOUNDARY = new VolumeConstraint();

MeshTopo::MeshTopo() :
    snapToBoundary(NO_BOUNDARY)
{
}

MeshTopo::MeshTopo(const glm::dvec3 &fixedPosition) :
    snapToBoundary(new VertexConstraint(-1, fixedPosition))
{
}

MeshTopo::MeshTopo(const AbstractConstraint* constraint) :
    snapToBoundary(constraint)
{
}

Mesh::Mesh() :
    _nodeGroups(new NodeGroups()),
    _boundary(new BoundaryFree())
{

}

Mesh::Mesh(const Mesh& m) :
    modelName(m.modelName),
    verts(m.verts),
    topos(m.topos),
    tets(m.tets),
    pyrs(m.pyrs),
    pris(m.pris),
    hexs(m.hexs),
    _nodeGroups(new NodeGroups(m.nodeGroups())),
    _boundary(m._boundary)
{

}

Mesh::~Mesh()
{

}

Mesh& Mesh::operator=(const Mesh& mesh)
{
    modelName = mesh.modelName;

    verts = mesh.verts;
    topos = mesh.topos;
    tets = mesh.tets;
    pyrs = mesh.pyrs;
    pris = mesh.pris;
    hexs = mesh.hexs;

    _nodeGroups.reset(new NodeGroups(mesh.nodeGroups()));
    _boundary = mesh._boundary;
}

void Mesh::clear()
{
    verts.clear();
    verts.shrink_to_fit();
    tets.clear();
    tets.shrink_to_fit();
    pyrs.clear();
    pyrs.shrink_to_fit();
    pris.clear();
    pris.shrink_to_fit();
    hexs.clear();
    hexs.shrink_to_fit();
    topos.clear();
    topos.shrink_to_fit();
    nodeGroups().clear();
}

void Mesh::compileTopology(bool verbose)
{
    if(verbose)
    {
        getLog().postMessage(new Message('I', false,
            modelName + ": Compiling mesh topology...", "Mesh"));
    }

    // Compact verts and elems data structures
    verts.shrink_to_fit();
    tets.shrink_to_fit();
    pyrs.shrink_to_fit();
    pris.shrink_to_fit();
    hexs.shrink_to_fit();

    size_t vertCount = verts.size();

    auto neigBegin = chrono::high_resolution_clock::now();
    topos.resize(vertCount);
    topos.shrink_to_fit();
    compileNeighborhoods();

    auto indeBegin = chrono::high_resolution_clock::now();

    nodeGroups().build(*this);

    auto compileEnd = chrono::high_resolution_clock::now();


    getLog().postMessage(new Message('I', false,
        "Vertice count: " + to_string(vertCount), "Mesh"));

    size_t elemCount = tets.size() + pris.size() + hexs.size();
    getLog().postMessage(new Message('I', false,
        "Element count: " + to_string(elemCount), "Mesh"));

    if(verbose)
    {
        double elemVertRatio = elemCount  / (double) vertCount;
        getLog().postMessage(new Message('I', false,
            "Element count / Vertice count: " + to_string(elemVertRatio), "Mesh"));

        getLog().postMessage(new Message('I', false,
            "Independent set count: " + to_string(nodeGroups().count()), "Mesh"));

        size_t neigVertCount = 0;
        size_t neigElemCount = 0;
        for(int i=0; i < vertCount; ++i)
        {
            neigVertCount += topos[i].neighborVerts.size();
            neigElemCount += topos[i].neighborElems.size();
        }

        int64_t meshMemorySize =
                int64_t(verts.size() * sizeof(decltype(verts.front()))) +
                int64_t(tets.size() * sizeof(decltype(tets.front()))) +
                int64_t(pris.size() * sizeof(decltype(pris.front()))) +
                int64_t(hexs.size() * sizeof(decltype(hexs.front()))) +
                int64_t(topos.size() * sizeof(decltype(topos.front()))) +
                int64_t(neigVertCount * sizeof(MeshNeigVert)) +
                int64_t(neigElemCount * sizeof(MeshNeigElem));
        getLog().postMessage(new Message('I', false,
            "Approx mesh size in memory: " + to_string(meshMemorySize) + " Bytes", "Mesh"));

        int neigTime = chrono::duration_cast<chrono::milliseconds>(indeBegin - neigBegin).count();
        getLog().postMessage(new Message('I', false,
            "Neighborhood compilation time: " + to_string(neigTime) + "ms", "Mesh"));

        int indeTime = chrono::duration_cast<chrono::milliseconds>(compileEnd - indeBegin).count();
        getLog().postMessage(new Message('I', false,
            "Independent groups compilation time: " + to_string(indeTime) + "ms", "Mesh"));
    }
}

void Mesh::updateGlslTopology() const
{

}

void Mesh::updateGlslVertices() const
{

}

void Mesh::fetchGlslVertices()
{

}

void Mesh::clearGlslMemory() const
{

}

void Mesh::updateCudaTopology() const
{

}

void Mesh::updateCudaVertices() const
{

}

void Mesh::fetchCudaVertices()
{

}

void Mesh::clearCudaMemory() const
{

}

std::string Mesh::meshGeometryShaderName() const
{
    return std::string();
}

unsigned int Mesh::glBuffer(const EMeshBuffer&) const
{
    return 0;
}

unsigned int Mesh::glBufferBinding(EBufferBinding binding) const
{
    return 0;
}

void Mesh::bindGlShaderStorageBuffers() const
{

}

void Mesh::printPropperties(OptimizationPlot& plot) const
{
    plot.addMeshProperty("Model Name",        modelName);
    plot.addMeshProperty("Vertex Count",      to_string(verts.size()));
    plot.addMeshProperty("Tet Count",         to_string(tets.size()));
    plot.addMeshProperty("Pyramid Count",     to_string(pyrs.size()));
    plot.addMeshProperty("Prism Count",       to_string(pris.size()));
    plot.addMeshProperty("Hex Count",         to_string(hexs.size()));
    plot.addMeshProperty("Patch Group Count", to_string(nodeGroups().count()));
}

void Mesh::setBoundary(const std::shared_ptr<AbstractBoundary>& boundary)
{
    _boundary = boundary;

    assert(_boundary->unitTest());
}

void Mesh::compileNeighborhoods()
{
    for(MeshTopo& topo : topos)
    {
        topo.neighborElems.clear();
        topo.neighborVerts.clear();
    }

    size_t tetCount = tets.size();
    for(size_t i=0; i < tetCount; ++i)
    {
        for(int v=0; v < MeshTet::VERTEX_COUNT; ++v)
        {
            topos[tets[i].v[v]].neighborElems.push_back(
                MeshNeigElem(i, MeshTet::ELEMENT_TYPE, v));
        }

        for(int e=0; e < MeshTet::EDGE_COUNT; ++e)
        {
            addEdge(tets[i].v[MeshTet::edges[e][0]],
                    tets[i].v[MeshTet::edges[e][1]]);
        }
    }

    size_t pyramidCount = pyrs.size();
    for(size_t i=0; i < pyramidCount; ++i)
    {
        for(int v=0; v < MeshPyr::VERTEX_COUNT; ++v)
        {
            topos[pyrs[i].v[v]].neighborElems.push_back(
                MeshNeigElem(i, MeshPyr::ELEMENT_TYPE, v));
        }

        for(int e=0; e < MeshPyr::EDGE_COUNT; ++e)
        {
            addEdge(pyrs[i].v[MeshPyr::edges[e][0]],
                    pyrs[i].v[MeshPyr::edges[e][1]]);
        }
    }

    size_t prismCount = pris.size();
    for(size_t i=0; i < prismCount; ++i)
    {
        for(int v=0; v < MeshPri::VERTEX_COUNT; ++v)
        {
            topos[pris[i].v[v]].neighborElems.push_back(
                MeshNeigElem(i, MeshPri::ELEMENT_TYPE, v));
        }

        for(int e=0; e < MeshPri::EDGE_COUNT; ++e)
        {
            addEdge(pris[i].v[MeshPri::edges[e][0]],
                    pris[i].v[MeshPri::edges[e][1]]);
        }
    }

    size_t hexCount = hexs.size();
    for(size_t i=0; i < hexCount; ++i)
    {
        for(int v=0; v < MeshHex::VERTEX_COUNT; ++v)
        {
            topos[hexs[i].v[v]].neighborElems.push_back(
                MeshNeigElem(i, MeshHex::ELEMENT_TYPE, v));
        }

        for(int e=0; e < MeshHex::EDGE_COUNT; ++e)
        {
            addEdge(hexs[i].v[MeshHex::edges[e][0]],
                    hexs[i].v[MeshHex::edges[e][1]]);
        }
    }


    // Compact topology neighborhoods
    size_t vertCount = verts.size();
    for(int i=0; i < vertCount; ++i)
    {
        topos[i].neighborVerts.shrink_to_fit();
        topos[i].neighborElems.shrink_to_fit();
    }
}

void Mesh::addEdge(int firstVert, int secondVert)
{
    vector<MeshNeigVert>& neighbors = topos[firstVert].neighborVerts;
    int neighborCount = neighbors.size();
    for(int n=0; n < neighborCount; ++n)
    {
        if(secondVert == neighbors[n])
            return;
    }

    // This really is a new edge
    topos[firstVert].neighborVerts.push_back(secondVert);
    topos[secondVert].neighborVerts.push_back(firstVert);
}
