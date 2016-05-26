#ifndef GPUMESH_BOUNDARY
#define GPUMESH_BOUNDARY

#include <vector>

#include "Constraints/Constraint.h"
#include "Constraints/VertexConstraint.h"
#include "Constraints/EdgeConstraint.h"
#include "Constraints/SurfaceConstraint.h"
#include "Constraints/VolumeConstraint.h"


class MeshBoundary
{
public:
    MeshBoundary();
    virtual ~MeshBoundary();

    const TopologyConstraint* volume() const;

    virtual const TopologyConstraint* merge(
        const TopologyConstraint* c1,
        const TopologyConstraint* c2) const = 0;

    virtual const TopologyConstraint* split(
        const TopologyConstraint* c1,
        const TopologyConstraint* c2) const = 0;

protected:
    void addSurface(const SurfaceConstraint* surface);
    void addEdge(const EdgeConstraint* edge);
    void addVertex(const VertexConstraint* vertex);


private:
    VolumeConstraint _volume;
    std::vector<const SurfaceConstraint*> _surface;
    std::vector<const EdgeConstraint*> _edges;
    std::vector<const VertexConstraint*> _vertices;
};



// IMPLEMENTATION //
inline const TopologyConstraint* MeshBoundary::volume() const
{
    return &_volume;
}

#endif // GPUMESH_BOUNDARY
