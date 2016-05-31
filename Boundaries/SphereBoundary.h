#ifndef GPUMESH_SPHERE_BOUNDARY
#define GPUMESH_SPHERE_BOUNDARY

#include "AbstractBoundary.h"


class SphereBoundary : public AbstractBoundary
{
public:
    SphereBoundary();
    virtual ~SphereBoundary();


    virtual bool unitTest() const override;


    const AbstractConstraint* face() const;


    static const double RADIUS;


private:
    class Face : public FaceConstraint
    {
    public:
        Face();
        virtual glm::dvec3 operator()(
            const glm::dvec3 &pos) const override;
    } _face;
};



// IMPLEMENTATION //
inline const AbstractConstraint* SphereBoundary::face() const
{
    return &_face;
}

#endif // GPUMESH_SPHERE_BOUNDARY
