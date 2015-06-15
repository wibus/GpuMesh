#ifndef GPUMESH_TETPOOL
#define GPUMESH_TETPOOL

#include <vector>

#include "Tetrahedron.h"


struct TetPool
{
    inline Tetrahedron* acquireTetrahedron(int v0, int v1, int v2, int v3)
    {
        if(_tetPool.empty())
        {
            return new Tetrahedron(v0, v1, v2, v3);
        }
        else
        {
            Tetrahedron* tet = _tetPool.back();
            _tetPool.pop_back();
            tet->v[0] = v0;
            tet->v[1] = v1;
            tet->v[2] = v2;
            tet->v[3] = v3;
            return tet;
        }
    }

    inline void disposeTetrahedron(Tetrahedron* tet)
    {
        _tetPool.push_back(tet);
    }

    inline void releaseMemoryPool()
    {
        int tetCount = _tetPool.size();
        for(int i=0; i < tetCount; ++i)
            delete _tetPool[i];

        _tetPool.clear();
        _tetPool.shrink_to_fit();
    }

private:
    std::vector<Tetrahedron*> _tetPool;
};

#endif // GPUMESH_TETPOOL
