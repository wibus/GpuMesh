#include "SurfaceConstraint.h"

#include "EdgeConstraint.h"


SurfaceConstraint::SurfaceConstraint(int id) :
    TopologyConstraint(id, 2),
    _volumes{nullptr, nullptr}
{
    assert(id > 0);
}

void SurfaceConstraint::addEdge(EdgeConstraint* edge)
{
    edge->addSurface(this);
    _edges.push_back(edge);
}

void SurfaceConstraint::addVolume(const VolumeConstraint* volume)
{
    if(_volumes[0] == nullptr)
        _volumes[0] = volume;
    else if(_volumes[1] == nullptr)
        _volumes[1] = volume;
}

bool SurfaceConstraint::isBoundedBy(const EdgeConstraint* edge) const
{
    for(const EdgeConstraint* e : _edges)
    {
        if(e == edge)
            return true;
    }

    return false;
}

const TopologyConstraint* SurfaceConstraint::split(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
}

const TopologyConstraint* SurfaceConstraint::merge(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
}


PlaneConstraint::PlaneConstraint(int id, const glm::dvec3 &p, const glm::dvec3 &n) :
    SurfaceConstraint(id),
    _p(p), _n(n)
{

}

glm::dvec3 PlaneConstraint::operator()(const glm::dvec3& pos) const
{
    glm::dvec3 dist = pos - _p;
    double l = glm::dot(dist, _n);
    return pos - _n * l;
}
