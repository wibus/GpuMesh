#ifndef GPUMESH_CPUMESHER
#define GPUMESH_CPUMESHER

#include "AbstractMesher.h"


class Adjacency;


class CpuMesher : public AbstractMesher
{
public:
    CpuMesher(Mesh& mesh, unsigned int vertCount);
    virtual ~CpuMesher() = 0;


protected:
    virtual void computeVertexLocations();
    virtual void clearVertexLocations();
    virtual void addEdge(int firstVert,
                         int secondVert);

    virtual void smoothMesh() override;


protected:
    bool _locationsComputed;
    std::vector<Adjacency> _adjacency;
};

#endif // GPUMESH_CPUMESHER
