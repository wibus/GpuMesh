#ifndef GPUMESH_VOLUME_CONSTRAINT
#define GPUMESH_VOLUME_CONSTRAINT

#include <vector>

#include "AbstractConstraint.h"


class VolumeConstraint : public AbstractConstraint
{
public:
    VolumeConstraint();

    void addSurface(SurfaceConstraint* surface);
    bool isBoundedBy(const SurfaceConstraint* surface) const;

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override;

protected:
    virtual const AbstractConstraint* split(const AbstractConstraint* c) const override;
    virtual const AbstractConstraint* merge(const AbstractConstraint* c) const override;

private:
    std::vector<SurfaceConstraint*> _surfaces;
};


#endif // GPUMESH_VOLUME_CONSTRAINT
