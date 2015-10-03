#include "KdTreeDiscretizer.h"

#include "DataStructures/GpuMesh.h"


KdTreeDiscretizer::KdTreeDiscretizer() :
    _gridMesh(new GpuMesh())
{

}

KdTreeDiscretizer::~KdTreeDiscretizer()
{

}

std::shared_ptr<Mesh> KdTreeDiscretizer::gridMesh() const
{
    return _gridMesh;
}

void KdTreeDiscretizer::discretize(const Mesh& mesh, const glm::ivec3& gridSize)
{

}
