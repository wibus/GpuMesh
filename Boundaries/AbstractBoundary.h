#ifndef GPUMESH_ABSTRACT_BOUNDARY
#define GPUMESH_ABSTRACT_BOUNDARY

#include <vector>

#include "Constraints/VertexConstraint.h"
#include "Constraints/EdgeConstraint.h"
#include "Constraints/SurfaceConstraint.h"
#include "Constraints/VolumeConstraint.h"


class AbstractBoundary
{
protected:
    AbstractBoundary();

public:
    virtual ~AbstractBoundary();

    const VolumeConstraint* volume() const;

    virtual const AbstractConstraint* split(
        const AbstractConstraint* c1,
        const AbstractConstraint* c2) const;

    virtual const AbstractConstraint* merge(
        const AbstractConstraint* c1,
        const AbstractConstraint* c2) const;

    static const AbstractConstraint* INVALID_OPERATION;

protected:
    VolumeConstraint* volume();
    void addSurface(const SurfaceConstraint* surface);
    void addEdge(const EdgeConstraint* edge);
    void addVertex(const VertexConstraint* vertex);


private:
    VolumeConstraint _volume;
};



// IMPLEMENTATION //
inline const VolumeConstraint* AbstractBoundary::volume() const
{
    return &_volume;
}

inline VolumeConstraint* AbstractBoundary::volume()
{
    return &_volume;
}


#endif // GPUMESH_ABSTRACT_BOUNDARY
