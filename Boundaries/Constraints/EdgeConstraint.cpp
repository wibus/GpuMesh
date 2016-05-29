#include "EdgeConstraint.h"

#include "VertexConstraint.h"
#include "SurfaceConstraint.h"


EdgeConstraint::EdgeConstraint(int id) :
    AbstractConstraint(id, 1),
    _vertices{nullptr, nullptr},
    _surfaces{nullptr, nullptr}
{
    assert(id > 0);
}

void EdgeConstraint::addVertex(VertexConstraint *vertex)
{
    if(!isBoundedBy(vertex))
    {
        if(_vertices[0] == nullptr)
            _vertices[0] = vertex;
        else if(_vertices[1] == nullptr)
            _vertices[1] = vertex;
        else
            assert(false);

        _surfaces[0]->addVertex(vertex);
        _surfaces[1]->addVertex(vertex);

        vertex->addEdge(this);
    }
}

bool EdgeConstraint::isBoundedBy(const VertexConstraint* vertex) const
{
    return vertex == _vertices[0] || vertex == _vertices[1];
}

void EdgeConstraint::addSurface(SurfaceConstraint* surface)
{
    if(!isBoundedBy(surface))
    {
        if(_surfaces[0] == nullptr)
            _surfaces[0] = surface;
        else if(_surfaces[1] == nullptr)
            _surfaces[1] = surface;
        else
            assert(false);

        _vertices[0]->addSurface(surface);
        _vertices[1]->addSurface(surface);

        surface->addEdge(this);
    }
}

bool EdgeConstraint::isBoundedBy(const SurfaceConstraint* surface) const
{
    return surface == _surfaces[0] || surface == _surfaces[1];
}

const AbstractConstraint* EdgeConstraint::split(const AbstractConstraint* c) const
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

        for(const SurfaceConstraint* surface : _surfaces)
            if(surface->isBoundedBy(e))
                return surface;

        return SPLIT_VOLUME;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        if(isBoundedBy(v))
            return v;

        for(const SurfaceConstraint* surface : _surfaces)
            if(surface->isBoundedBy(v))
                return surface;

        return SPLIT_VOLUME;
    }

    return SPLIT_VOLUME;
}

const AbstractConstraint* EdgeConstraint::merge(const AbstractConstraint* c) const
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

        if(_vertices[0]->isBoundedBy(s))
            return _vertices[0];
        else if(_vertices[1]->isBoundedBy(s))
            return _vertices[1];

        return MERGE_PREVENT;
    }
    else if(c->dimension() == 1)
    {
        const EdgeConstraint* e =
            static_cast<const EdgeConstraint*>(c);

        if(e->isBoundedBy(_vertices[0]))
            return _vertices[0];
        else if(e->isBoundedBy(_vertices[1]))
                return _vertices[1];

        return MERGE_PREVENT;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        if(isBoundedBy(v))
            return v;
    }

    return MERGE_PREVENT;
}


SegmentConstraint::SegmentConstraint(int id, const glm::dvec3 &a, const glm::dvec3 &b) :
    EdgeConstraint(id),
    _a(a), _b(b), _u(glm::normalize(b-a)),
    _length(glm::distance(a, b))
{

}

glm::dvec3 SegmentConstraint::operator()(const glm::dvec3& pos) const
{
    glm::dvec3 dist = pos - _a;
    double l = glm::dot(dist, _u);

    if(l < 0)
        return _a;
    else if(l > _length)
        return _b;
    else
    {
        glm::dvec3 t = dist - _u * l;
        return pos - t;
    }
}
