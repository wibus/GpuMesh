#ifndef GPUMESH_SHELL_BOUNDARY
#define GPUMESH_SHELL_BOUNDARY

#include "AbstractBoundary.h"


class ShellBoundary : public AbstractBoundary
{
public:
    ShellBoundary();
    virtual ~ShellBoundary();


    const AbstractConstraint* inSurface() const;

    const AbstractConstraint* outSurface() const;

    const AbstractConstraint* volume() const;

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
inline const AbstractConstraint* ShellBoundary::inSurface() const
{
    return &_inSurface;
}

inline const AbstractConstraint* ShellBoundary::outSurface() const
{
    return &_outSurface;
}

#endif // GPUMESH_SHELL_BOUNDARY
