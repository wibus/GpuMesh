#ifndef GPUMESH_SPHERE_BOUNDARY
#define GPUMESH_SPHERE_BOUNDARY

#include "Boundary.h"


class SphereBoundary : public MeshBoundary
{
public:
    SphereBoundary();
    virtual ~SphereBoundary();


    const TopologyConstraint* surface() const;

private:
    class Surface : public SurfaceConstraint
    {
    public:
        Surface();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _surface;
};



// IMPLEMENTATION //
inline const TopologyConstraint* SphereBoundary::surface() const
{
    return &_surface;
}

#endif // GPUMESH_SPHERE_BOUNDARY
