#include "VertexConstraint.h"


VertexConstraint::VertexConstraint(int id, const glm::dvec3 position) :
    TopologyConstraint(id, 0),
    _pos(position)
{
    assert(id < 0);
}

void VertexConstraint::addEdge(const EdgeConstraint* edge)
{
    _edges.push_back(edge);
}

glm::dvec3 VertexConstraint::operator()(const glm::dvec3& pos) const
{
    return _pos;
}

const TopologyConstraint* VertexConstraint::split(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
}

const TopologyConstraint* VertexConstraint::merge(const TopologyConstraint* c) const
{
    // TODO
    return nullptr;
}
