#include "Boundary.h"

#include "Constraints/VertexConstraint.h"
#include "Constraints/EdgeConstraint.h"
#include "Constraints/SurfaceConstraint.h"
#include "Constraints/VolumeConstraint.h"


MeshBoundary::MeshBoundary()
{

}

MeshBoundary::~MeshBoundary()
{

}


const TopologyConstraint* MeshBoundary::merge(
    const TopologyConstraint* c1,
    const TopologyConstraint* c2) const
{
    if(c1 == c2)
        return c1;

    return &_volume;
}

const TopologyConstraint* MeshBoundary::split(
    const TopologyConstraint* c1,
    const TopologyConstraint* c2) const
{
    if(c1 == c2)
        return c1;

    if(c1->dimension() == 3 || c2->dimension() == 3)
        return &_volume;

    return nullptr;
}
