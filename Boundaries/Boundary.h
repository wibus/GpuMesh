#ifndef GPUMESH_BOUNDARY
#define GPUMESH_BOUNDARY

#include <vector>

#include "Constraints/VertexConstraint.h"
#include "Constraints/EdgeConstraint.h"
#include "Constraints/SurfaceConstraint.h"
#include "Constraints/VolumeConstraint.h"


class MeshBoundary
{
public:
    MeshBoundary();
    virtual ~MeshBoundary();

    const VolumeConstraint* volume() const;

    virtual const TopologyConstraint* merge(
        const TopologyConstraint* c1,
        const TopologyConstraint* c2) const = 0;

    virtual const TopologyConstraint* split(
        const TopologyConstraint* c1,
        const TopologyConstraint* c2) const = 0;

protected:
    VolumeConstraint* volume();
    void addSurface(const SurfaceConstraint* surface);
    void addEdge(const EdgeConstraint* edge);
    void addVertex(const VertexConstraint* vertex);


private:
    VolumeConstraint _volume;
};



// IMPLEMENTATION //
inline const VolumeConstraint* MeshBoundary::volume() const
{
    return &_volume;
}

inline VolumeConstraint* MeshBoundary::volume()
{
    return &_volume;
}


#endif // GPUMESH_BOUNDARY
