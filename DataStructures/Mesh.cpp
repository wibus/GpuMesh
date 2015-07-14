#include "Mesh.h"

#include <algorithm>
#include <iostream>

#include <CellarWorkbench/Misc/Log.h>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;
using namespace cellar;


const MeshEdge MeshTet::edges[MeshTet::EDGE_COUNT] = {
    MeshEdge(0, 1),
    MeshEdge(0, 2),
    MeshEdge(0, 3),
    MeshEdge(1, 2),
    MeshEdge(2, 3),
    MeshEdge(3, 1)
};

const MeshTri MeshTet::tris[MeshTet::TRI_COUNT] = {
    MeshTri(0, 1, 2),
    MeshTri(0, 2, 3),
    MeshTri(0, 3, 1),
    MeshTri(1, 3, 2)
};

const MeshTet MeshTet::tets[MeshTet::TET_COUNT] = {
    MeshTet(0, 1, 2, 3)
};


const MeshEdge MeshPri::edges[MeshPri::EDGE_COUNT] = {
    MeshEdge(0, 1),
    MeshEdge(0, 2),
    MeshEdge(1, 3),
    MeshEdge(2, 3),
    MeshEdge(0, 4),
    MeshEdge(1, 5),
    MeshEdge(2, 4),
    MeshEdge(3, 5),
    MeshEdge(4, 5)
};

const MeshTri MeshPri::tris[MeshPri::TRI_COUNT] = {
    MeshTri(2, 1, 0), // Z neg face 0
    MeshTri(1, 2, 3), // Z neg face 1
    MeshTri(1, 4, 0), // Y neg face 0
    MeshTri(4, 1, 5), // Y neg face 1
    MeshTri(4, 3, 2), // Y pos face 0
    MeshTri(3, 4, 5), // Y pos face 1
    MeshTri(0, 4, 2), // X neg face
    MeshTri(1, 3, 5)  // X pos face
};

const MeshTet MeshPri::tets[MeshPri::TET_COUNT] = {
    MeshTet(4, 0, 1, 2),
    MeshTet(5, 1, 3, 2),
    MeshTet(4, 1, 5, 2)
};


const MeshEdge MeshHex::edges[MeshHex::EDGE_COUNT] = {
    MeshEdge(0, 1),
    MeshEdge(0, 2),
    MeshEdge(1, 3),
    MeshEdge(2, 3),
    MeshEdge(0, 4),
    MeshEdge(1, 5),
    MeshEdge(2, 6),
    MeshEdge(3, 7),
    MeshEdge(4, 5),
    MeshEdge(4, 6),
    MeshEdge(5, 7),
    MeshEdge(6, 7)
};

const MeshTri MeshHex::tris[MeshHex::TRI_COUNT] = {
    MeshTri(2, 1, 0), // Z neg face 0
    MeshTri(1, 2, 3), // Z pos face 1
    MeshTri(5, 6, 4), // Z pos face 0
    MeshTri(6, 5, 7), // Z pos face 1
    MeshTri(1, 4, 0), // Y neg face 0
    MeshTri(4, 1, 5), // Y neg face 1
    MeshTri(2, 7, 3), // Y pos face 0
    MeshTri(7, 2, 6), // Y pos face 1
    MeshTri(4, 2, 0), // X neg face 0
    MeshTri(2, 4, 6), // X neg face 1
    MeshTri(7, 1, 3), // X pos face 0
    MeshTri(1, 7, 5), // X pos face 1
};

const MeshTet MeshHex::tets[MeshHex::TET_COUNT] = {
    MeshTet(0, 1, 4, 2),
    MeshTet(3, 1, 7, 2),
    MeshTet(5, 1, 7, 4),
    MeshTet(6, 2, 4, 7),
    MeshTet(1, 7, 4, 2)
};


MeshBound::MeshBound(int id) :
    _id(id)
{

}

MeshBound::~MeshBound()
{

}

glm::dvec3 MeshBound::operator()(const glm::dvec3& pos) const
{
    return pos;
}


const MeshBound MeshTopo::NO_BOUNDARY = MeshBound(0);

MeshTopo::MeshTopo() :
    isFixed(false),
    isBoundary(false),
    boundaryCallback(NO_BOUNDARY)
{
}

MeshTopo::MeshTopo(
        bool isFixed) :
    isFixed(isFixed),
    isBoundary(false),
    boundaryCallback(NO_BOUNDARY)
{
}

MeshTopo::MeshTopo(
        const MeshBound& boundaryCallback) :
    isFixed(false),
    isBoundary(&boundaryCallback != &NO_BOUNDARY),
    boundaryCallback(boundaryCallback)
{
}


Mesh::Mesh()
{

}

Mesh::~Mesh()
{

}

void Mesh::clear()
{
    vert.clear();
    tetra.clear();
    prism.clear();
    hexa.clear();
    topo.clear();
}

void Mesh::compileTopoly()
{
    getLog().postMessage(new Message('I', false,
        "Compiling mesh topology", "Mesh"));

    int vertCount = vert.size();

    topo.resize(vertCount);
    topo.shrink_to_fit();

    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        for(int v=0; v < MeshTet::VERTEX_COUNT; ++v)
        {
            topo[tetra[i].v[v]].neighborElems.push_back(
                MeshNeigElem(MeshTet::ELEMENT_TYPE, i));
        }

        for(int e=0; e < MeshTet::EDGE_COUNT; ++e)
        {
            addEdge(tetra[i].v[MeshTet::edges[e][0]],
                    tetra[i].v[MeshTet::edges[e][1]]);
        }
    }

    int prismCount = prism.size();
    for(int i=0; i < prismCount; ++i)
    {
        for(int v=0; v < MeshPri::VERTEX_COUNT; ++v)
        {
            topo[prism[i].v[v]].neighborElems.push_back(
                MeshNeigElem(MeshPri::ELEMENT_TYPE, i));
        }

        for(int e=0; e < MeshPri::EDGE_COUNT; ++e)
        {
            addEdge(prism[i].v[MeshPri::edges[e][0]],
                    prism[i].v[MeshPri::edges[e][1]]);
        }
    }

    int hexCount = hexa.size();
    for(int i=0; i < hexCount; ++i)
    {
        for(int v=0; v < MeshHex::VERTEX_COUNT; ++v)
        {
            topo[hexa[i].v[v]].neighborElems.push_back(
                MeshNeigElem(MeshHex::ELEMENT_TYPE, i));
        }

        for(int e=0; e < MeshHex::EDGE_COUNT; ++e)
        {
            addEdge(hexa[i].v[MeshHex::edges[e][0]],
                    hexa[i].v[MeshHex::edges[e][1]]);
        }
    }

    for(int i=0; i < vertCount; ++i)
    {
        topo[i].neighborVerts.shrink_to_fit();
        topo[i].neighborElems.shrink_to_fit();
    }
}

void Mesh::updateGpuTopoly()
{

}

void Mesh::updateGpuVertices()
{

}

void Mesh::updateCpuVertices()
{

}

std::string Mesh::meshGeometryShaderName() const
{
    return std::string();
}

void Mesh::uploadGeometry(cellar::GlProgram& program) const
{

}

unsigned int Mesh::glBuffer(const EMeshBuffer&) const
{
    return 0;
}

void Mesh::bindShaderStorageBuffers() const
{

}

size_t Mesh::firstFreeBufferBinding() const
{
    return 0;
}

void Mesh::addEdge(int firstVert, int secondVert)
{
    vector<MeshNeigVert>& neighbors = topo[firstVert].neighborVerts;
    int neighborCount = neighbors.size();
    for(int n=0; n < neighborCount; ++n)
    {
        if(secondVert == neighbors[n])
            return;
    }

    // This really is a new edge
    topo[firstVert].neighborVerts.push_back(secondVert);
    topo[secondVert].neighborVerts.push_back(firstVert);
}
