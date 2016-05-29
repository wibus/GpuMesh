#include "VertexConstraint.h"

#include "EdgeConstraint.h"
#include "SurfaceConstraint.h"


VertexConstraint::VertexConstraint(int id, const glm::dvec3 position) :
    AbstractConstraint(id, 0),
    _pos(position)
{
    assert(id < 0);
}

void VertexConstraint::addEdge(EdgeConstraint* edge)
{
    if(!isBoundedBy(edge))
    {
        _edges.push_back(edge);

        edge->addVertex(this);
    }
}

bool VertexConstraint::isBoundedBy(const EdgeConstraint* edge) const
{
    for(const EdgeConstraint* e : _edges)
        if(e == edge)
            return true;

    return false;
}

void VertexConstraint::addSurface(SurfaceConstraint* surface)
{
    if(!isBoundedBy(surface))
    {
        _surfaces.push_back(surface);

        surface->addVertex(this);
    }
}

bool VertexConstraint::isBoundedBy(const SurfaceConstraint* surface) const
{
    for(const SurfaceConstraint* s : _surfaces)
        if(s == surface)
            return true;

    return false;
}

glm::dvec3 VertexConstraint::operator()(const glm::dvec3& pos) const
{
    return _pos;
}

const AbstractConstraint* VertexConstraint::split(const AbstractConstraint* c) const
{
    if(c->dimension() == 3)
    {
        return c;
    }
    else if(c->dimension() == 2)
    {
        const SurfaceConstraint* s =
            static_cast<const SurfaceConstraint*>(c);

        if(isBoundedBy(s))
            return s;

        return SPLIT_VOLUME;
    }
    else if(c->dimension() == 1)
    {
        const EdgeConstraint* e =
            static_cast<const EdgeConstraint*>(c);

        if(isBoundedBy(e))
            return e;

        for(const SurfaceConstraint* surface : _surfaces)
            if(surface->isBoundedBy(e))
                return surface;

        return SPLIT_VOLUME;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        for(const EdgeConstraint* edge : _edges)
            if(edge->isBoundedBy(v))
                return edge;

        for(const SurfaceConstraint* surface : _surfaces)
            if(surface->isBoundedBy(v))
                return surface;

        return SPLIT_VOLUME;
    }

    return SPLIT_VOLUME;
}

const AbstractConstraint* VertexConstraint::merge(const AbstractConstraint* c) const
{
    if(c->dimension() == 3)
    {
        return this;
    }
    else if(c->dimension() == 2)
    {
        const SurfaceConstraint* s =
            static_cast<const SurfaceConstraint*>(c);

        if(isBoundedBy(s))
            return this;

        return MERGE_PREVENT;
    }
    else if(c->dimension() == 1)
    {
        const EdgeConstraint* e =
            static_cast<const EdgeConstraint*>(c);

        if(isBoundedBy(e))
            return this;

        return MERGE_PREVENT;
    }
    else if(c->dimension() == 0)
    {
        return MERGE_PREVENT;
    }

    return MERGE_PREVENT;
}
