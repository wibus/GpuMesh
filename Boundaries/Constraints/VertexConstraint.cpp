#include "VertexConstraint.h"

#include "EdgeConstraint.h"
#include "FaceConstraint.h"


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

void VertexConstraint::addFace(FaceConstraint* face)
{
    if(!isBoundedBy(face))
    {
        _faces.push_back(face);

        face->addVertex(this);
    }
}

bool VertexConstraint::isBoundedBy(const FaceConstraint* face) const
{
    for(const FaceConstraint* s : _faces)
        if(s == face)
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
        const FaceConstraint* s =
            static_cast<const FaceConstraint*>(c);

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

        for(const FaceConstraint* face : _faces)
            if(face->isBoundedBy(e))
                return face;

        return SPLIT_VOLUME;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        for(const EdgeConstraint* edge : _edges)
            if(edge->isBoundedBy(v))
                return edge;

        for(const FaceConstraint* face : _faces)
            if(face->isBoundedBy(v))
                return face;

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
        const FaceConstraint* s =
            static_cast<const FaceConstraint*>(c);

        if(isBoundedBy(s))
            return this;

        return PREVENT_MERGE;
    }
    else if(c->dimension() == 1)
    {
        const EdgeConstraint* e =
            static_cast<const EdgeConstraint*>(c);

        if(isBoundedBy(e))
            return this;

        return PREVENT_MERGE;
    }
    else if(c->dimension() == 0)
    {
        return PREVENT_MERGE;
    }

    return PREVENT_MERGE;
}
