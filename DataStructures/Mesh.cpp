#include "Mesh.h"

#include <algorithm>
#include <iostream>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;


const MeshTri MeshTet::faces[MeshTet::FACE_COUNT] = {
    MeshTri(0, 1, 2),
    MeshTri(0, 2, 3),
    MeshTri(0, 3, 1),
    MeshTri(1, 3, 2)
};

const int MeshTet::edges[MeshTet::EDGE_COUNT][2] = {
    {0, 1},
    {0, 2},
    {0, 3},
    {1, 2},
    {2, 3},
    {3, 1}
};


const MeshTri MeshPri::faces[MeshPri::FACE_COUNT] = {
    MeshTri(2, 1, 0), // Z neg face 0
    MeshTri(1, 2, 3), // Z neg face 1
    MeshTri(1, 4, 0), // Y neg face 0
    MeshTri(4, 1, 5), // Y neg face 1
    MeshTri(4, 3, 2), // Y pos face 0
    MeshTri(3, 4, 5), // Y pos face 1
    MeshTri(0, 4, 2), // X neg face
    MeshTri(1, 3, 5)  // X pos face
};

const int MeshPri::edges[MeshPri::EDGE_COUNT][2] = {
    {0, 1},
    {0, 2},
    {1, 3},
    {2, 3},
    {0, 4},
    {1, 5},
    {2, 4},
    {3, 5},
    {4, 5}
};


const MeshTri MeshHex::faces[MeshHex::FACE_COUNT] = {
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

const int MeshHex::edges[MeshHex::EDGE_COUNT][2] = {
    {0, 1},
    {0, 2},
    {1, 3},
    {2, 3},
    {0, 4},
    {1, 5},
    {2, 6},
    {3, 7},
    {4, 5},
    {4, 6},
    {5, 7},
    {6, 7}
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
    int vertCount = vert.size();

    topo.resize(vertCount);
    topo.shrink_to_fit();

    int tetCount = tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        for(int e=0; e < MeshTet::EDGE_COUNT; ++e)
        {
            addEdge(tetra[i].v[MeshTet::edges[e][0]],
                    tetra[i].v[MeshTet::edges[e][1]]);
        }
    }

    int prismCount = prism.size();
    for(int i=0; i < prismCount; ++i)
    {
        for(int e=0; e < MeshPri::EDGE_COUNT; ++e)
        {
            addEdge(prism[i].v[MeshPri::edges[e][0]],
                    prism[i].v[MeshPri::edges[e][1]]);
        }
    }

    int hexCount = hexa.size();
    for(int i=0; i < hexCount; ++i)
    {
        for(int e=0; e < MeshHex::EDGE_COUNT; ++e)
        {
            addEdge(hexa[i].v[MeshHex::edges[e][0]],
                    hexa[i].v[MeshHex::edges[e][1]]);
        }
    }

    for(int i=0; i < vertCount; ++i)
    {
        topo[i].neighbors.shrink_to_fit();
    }
}

unsigned int Mesh::glBuffer(const EMeshBuffer&) const
{
    return 0;
}

void Mesh::addEdge(int firstVert, int secondVert)
{
    vector<int>& neighbors = topo[firstVert].neighbors;
    int neighborCount = neighbors.size();
    for(int n=0; n < neighborCount; ++n)
    {
        if(secondVert == neighbors[n])
            return;
    }

    // This really is a new edge
    topo[firstVert].neighbors.push_back(secondVert);
    topo[secondVert].neighbors.push_back(firstVert);
}
