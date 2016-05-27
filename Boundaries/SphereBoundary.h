#ifndef GPUMESH_SPHERE_BOUNDARY
#define GPUMESH_SPHERE_BOUNDARY

#include "AbstractBoundary.h"


class SphereBoundary : public AbstractBoundary
{
public:
    SphereBoundary();
    virtual ~SphereBoundary();


    const AbstractConstraint* surface() const;

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
inline const AbstractConstraint* SphereBoundary::surface() const
{
    return &_surface;
}

#endif // GPUMESH_SPHERE_BOUNDARY
