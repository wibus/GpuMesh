#ifndef GPUMESH_SHELL_BOUNDARY
#define GPUMESH_SHELL_BOUNDARY

#include "AbstractBoundary.h"


class ShellBoundary : public AbstractBoundary
{
public:
    ShellBoundary();
    virtual ~ShellBoundary();


    virtual bool unitTest() const override;


    const AbstractConstraint* inFace() const;

    const AbstractConstraint* outFace() const;


    static const double IN_RADIUS;
    static const double OUT_RADIUS;


private:
    class InFace : public FaceConstraint
    {
    public:
        InFace();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _inFace;

    class OutFace : public FaceConstraint
    {
    public:
        OutFace();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _outFace;
};



// IMPLEMENTATION //
inline const AbstractConstraint* ShellBoundary::inFace() const
{
    return &_inFace;
}

inline const AbstractConstraint* ShellBoundary::outFace() const
{
    return &_outFace;
}

#endif // GPUMESH_SHELL_BOUNDARY
