#include "SurfaceConstraint.h"

#include "VertexConstraint.h"
#include "EdgeConstraint.h"


SurfaceConstraint::SurfaceConstraint(int id) :
    AbstractConstraint(id, 2),
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
    else
        assert(false);
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

const AbstractConstraint* SurfaceConstraint::split(const AbstractConstraint* c) const
{
    if(c->dimension() == 3)
    {
        return c;
    }
    else if(c->dimension() == 2)
    {
        return SPLIT_VOLUME;
    }
    else if(c->dimension() == 1)
    {
        const EdgeConstraint* e =
            static_cast<const EdgeConstraint*>(c);

        if(isBoundedBy(e))
            return this;

        return SPLIT_VOLUME;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        for(const EdgeConstraint* edge : _edges)
            if(edge->isBoundedBy(v))
                return this;

        return SPLIT_VOLUME;
    }

    return SPLIT_VOLUME;
}

const AbstractConstraint* SurfaceConstraint::merge(const AbstractConstraint* c) const
{
    if(c->dimension() == 3)
    {
        return this;
    }
    else if(c->dimension() == 2)
    {
        const SurfaceConstraint* s =
            static_cast<const SurfaceConstraint*>(c);

        for(const EdgeConstraint* edge : _edges)
            if(edge->isBoundedBy(s))
                return edge;

        return MERGE_PREVENT;
    }
    else if(c->dimension() == 1)
    {
        const EdgeConstraint* e =
            static_cast<const EdgeConstraint*>(c);

        if(isBoundedBy(e))
            return e;

        return MERGE_PREVENT;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        for(const EdgeConstraint* edge : _edges)
            if(edge->isBoundedBy(v))
                return v;

        return MERGE_PREVENT;
    }

    return MERGE_PREVENT;
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
