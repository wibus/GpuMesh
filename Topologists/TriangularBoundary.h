#ifndef GPUMESH_TRIANGULARBOUNDARY
#define GPUMESH_TRIANGULARBOUNDARY

#include <map>

#include "DataStructures/Mesh.h"
#include "DataStructures/TriSet.h"


class TriangularBoundary
{
public:
    struct Edge
    {
        Edge(uint v0, uint v1);
        bool operator<(const Edge& e) const;
        uint v[2];
    };

    TriangularBoundary(const std::vector<MeshTet>& tets);

    void insertTet(const MeshTet& tet, uint tetId = APPEND);

    void removeTet(uint tetId);

    size_t tetCount() const;
    MeshLocalTet& tet(uint tetId);
    const MeshLocalTet& tet(uint tetId) const;
    bool isBoundary(uint vId, uint nId) const;

    static const uint APPEND;

protected:
    void insertTriEdges(const Triangle& tri);
    void removeTriEdges(const Triangle& tri);

private:
    TriSet _triSet;
    std::map<Edge, int> _edges;
    std::vector<MeshLocalTet> _tets;
};



// IMPLEMENTATION //
inline TriangularBoundary::Edge::Edge(uint v0, uint v1)
{
    if(v0 < v1)
        {v[0]=v0; v[1]=v1;}
    else
        {v[0]=v1; v[1]=v0;}
}

inline bool TriangularBoundary::Edge::operator<(const Edge& e) const
{
    if(v[0] < e.v[0])
        return true;
    else if(v[0] > e.v[0])
        return false;
    else
    {
        if(v[1] < e.v[1])
            return true;
        else
            return false;
    }
}

inline size_t TriangularBoundary::tetCount() const
{
    return _tets.size();
}

inline MeshLocalTet& TriangularBoundary::tet(uint tetId)
{
    return _tets[tetId];
}

inline const MeshLocalTet& TriangularBoundary::tet(uint tetId) const
{
    return _tets[tetId];
}

#endif // GPUMESH_TRIANGULARBOUNDARY
