#ifndef GPUMESH_SHELL_BOUNDARY
#define GPUMESH_SHELL_BOUNDARY

#include "Boundary.h"


class ShellBoundary : public MeshBoundary
{
public:
    ShellBoundary();
    virtual ~ShellBoundary();


    const TopologyConstraint* inSurface() const;

    const TopologyConstraint* outSurface() const;

    const TopologyConstraint* volume() const;

private:
    class InSurface : public SurfaceConstraint
    {
    public:
        InSurface();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _inSurface;

    class OutSurface : public SurfaceConstraint
    {
    public:
        OutSurface();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _outSurface;
};



// IMPLEMENTATION //
inline const TopologyConstraint* ShellBoundary::inSurface() const
{
    return &_inSurface;
}

inline const TopologyConstraint* ShellBoundary::outSurface() const
{
    return &_outSurface;
}

#endif // GPUMESH_SHELL_BOUNDARY
