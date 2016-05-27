#include "VolumeConstraint.h"

#include "SurfaceConstraint.h"


VolumeConstraint::VolumeConstraint() :
    AbstractConstraint(0, 3)
{

}

void VolumeConstraint::addSurface(SurfaceConstraint *surface)
{
    surface->addVolume(this);
    _surfaces.push_back(surface);
}

glm::dvec3 VolumeConstraint::operator()(const glm::dvec3& pos) const
{
    return pos;
}

const AbstractConstraint* VolumeConstraint::split(const AbstractConstraint* c) const
{
    return this;
}

const AbstractConstraint* VolumeConstraint::merge(const AbstractConstraint* c) const
{
    return c;
}
