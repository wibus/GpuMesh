#ifndef GPUMESH_VOLUME_CONSTRAINT
#define GPUMESH_VOLUME_CONSTRAINT

#include <vector>

#include "Constraint.h"


class VolumeConstraint : public TopologyConstraint
{
public:
    VolumeConstraint();

    void addSurface(SurfaceConstraint* surface);

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override;

    virtual const TopologyConstraint* split(const TopologyConstraint* c) const override;
    virtual const TopologyConstraint* merge(const TopologyConstraint* c) const override;

private:
    std::vector<const SurfaceConstraint*> _surfaces;
};


#endif // GPUMESH_VOLUME_CONSTRAINT
