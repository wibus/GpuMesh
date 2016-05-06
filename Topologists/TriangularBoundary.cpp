#include "TriangularBoundary.h"


const uint TriangularBoundary::APPEND = -1;


TriangularBoundary::TriangularBoundary(const std::vector<MeshTet>& tets)
{
    size_t tetCount = tets.size();
    size_t triCount = tetCount * 4;

    _tets.reserve(tets.size());
    _triSet.reset(glm::max(10.0,
        pow(double(triCount), 2.0/3.0)));

    for(size_t t=0; t < tetCount; ++t)
    {
        insertTet(tets[t], APPEND);
    }
}

void TriangularBoundary::insertTet(const MeshTet& tet, uint tetId)
{
    if(tetId == APPEND)
    {
        tetId = _tets.size();
        _tets.push_back(MeshLocalTet(tet));
    }
    else
    {
        _tets[tetId] = MeshLocalTet(tet);
    }

    MeshLocalTet& ltet = _tets[tetId];
    for(uint s=0; s < MeshTet::TRI_COUNT; ++s)
    {
        Triangle tri(ltet.v[MeshTet::tris[s][0]],
                     ltet.v[MeshTet::tris[s][1]],
                     ltet.v[MeshTet::tris[s][2]]);
        glm::uvec2 con = _triSet.xOrTri(tri, tetId, s);

        uint owner = con[0];
        uint side = con[1];
        ltet.n[s] = owner;

        if(owner != TriSet::NO_OWNER)
        {
            removeTriEdges(tri);
            _tets[owner].n[side] = tetId;
        }
        else
        {
            insertTriEdges(tri);
        }
    }
}

void TriangularBoundary::removeTet(uint tetId)
{
    const MeshLocalTet& tet = _tets[tetId];
    for(uint s=0; s < MeshTet::TRI_COUNT; ++s)
    {
        Triangle tri(tet.v[MeshTet::tris[s][0]],
                     tet.v[MeshTet::tris[s][1]],
                     tet.v[MeshTet::tris[s][2]]);

        uint side = 0;
        uint owner = tet.n[s];
        if(owner != TriSet::NO_OWNER)
        {
            insertTriEdges(tri);
            MeshLocalTet& ot = _tets[owner];
                 if(ot.n[0] == tetId) {side = 0; ot.n[0] = TriSet::NO_OWNER;}
            else if(ot.n[1] == tetId) {side = 1; ot.n[1] = TriSet::NO_OWNER;}
            else if(ot.n[2] == tetId) {side = 2; ot.n[2] = TriSet::NO_OWNER;}
            else if(ot.n[3] == tetId) {side = 3; ot.n[3] = TriSet::NO_OWNER;}
        }
        else
        {
            removeTriEdges(tri);
        }

        _triSet.xOrTri(tri, owner, side);
    }
}

bool TriangularBoundary::isBoundary(
        uint vId, uint nId) const
{
    return _edges.find(Edge(vId, nId)) != _edges.end();
}

void TriangularBoundary::insertTriEdges(const Triangle& tri)
{
    ++_edges[Edge(tri.v[0], tri.v[1])];
    ++_edges[Edge(tri.v[1], tri.v[2])];
    ++_edges[Edge(tri.v[0], tri.v[2])];
}

void TriangularBoundary::removeTriEdges(const Triangle& tri)
{
    Edge edges[] = {
        Edge(tri.v[0], tri.v[1]),
        Edge(tri.v[1], tri.v[2]),
        Edge(tri.v[2], tri.v[0])
    };

    for(int i=0; i < 3; ++i)
    {
        auto it = _edges.find(edges[i]);
        if(it != _edges.end())
        {
            --it->second;
            if(it->second <= 0)
                _edges.erase(it);
        }
    }
}
