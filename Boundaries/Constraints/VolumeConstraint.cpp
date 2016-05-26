#include "VolumeConstraint.h"

#include "SurfaceConstraint.h"


VolumeConstraint::VolumeConstraint() :
    TopologyConstraint(0, 3)
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

const TopologyConstraint* VolumeConstraint::split(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
}

const TopologyConstraint* VolumeConstraint::merge(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
}
