#include "EdgeConstraint.h"

#include "VertexConstraint.h"
#include "SurfaceConstraint.h"


EdgeConstraint::EdgeConstraint(int id) :
    TopologyConstraint(id, 1),
    _vertices{nullptr, nullptr},
    _surfaces{nullptr, nullptr}
{
    assert(id > 0);
}

void EdgeConstraint::addVertex(VertexConstraint *vertex)
{
    vertex->addEdge(this);

    if(_vertices[0] == nullptr)
        _vertices[0] = vertex;
    else if(_vertices[1] == nullptr)
        _vertices[1] = vertex;
}

bool EdgeConstraint::isBoundedBy(const VertexConstraint* v) const
{
    return v == _vertices[0] || v == _vertices[1];
}

void EdgeConstraint::addSurface(const SurfaceConstraint* surface)
{
    if(_surfaces[0] == nullptr)
        _surfaces[0] = surface;
    else if(_surfaces[1] == nullptr)
        _surfaces[1] = surface;
}

const SurfaceConstraint* EdgeConstraint::getSurface(const EdgeConstraint* edge)
{
    for(const SurfaceConstraint* surface : _surfaces)
    {
        if(surface->isBoundedBy(edge))
            return surface;
    }

    return nullptr;
}

const TopologyConstraint* EdgeConstraint::split(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
}

const TopologyConstraint* EdgeConstraint::merge(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
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
