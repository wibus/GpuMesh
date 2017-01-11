#include "FaceConstraint.h"

#include "VertexConstraint.h"
#include "EdgeConstraint.h"
#include "VolumeConstraint.h"


FaceConstraint::FaceConstraint(int id) :
    AbstractConstraint(id, 2),
    _volumes{nullptr, nullptr}
{
    assert(id > 0);
}

void FaceConstraint::addVertex(VertexConstraint* vertex)
{
    if(!isBoundedBy(vertex))
    {
        _vertices.push_back(vertex);

        vertex->addFace(this);
    }
}

bool FaceConstraint::isBoundedBy(const VertexConstraint* vertex) const
{
    for(const VertexConstraint* v : _vertices)
    {
        if(v == vertex)
            return true;
    }

    return false;
}

void FaceConstraint::addEdge(EdgeConstraint* edge)
{
    if(!isBoundedBy(edge))
    {
        _edges.push_back(edge);

        edge->addFace(this);
    }
}

bool FaceConstraint::isBoundedBy(const EdgeConstraint* edge) const
{
    for(const EdgeConstraint* e : _edges)
    {
        if(e == edge)
            return true;
    }

    return false;
}

void FaceConstraint::addVolume(VolumeConstraint* volume)
{
    if(!isBoundedBy(volume))
    {
        if(_volumes[0] == nullptr)
            _volumes[0] = volume;
        else if(_volumes[1] == nullptr)
            _volumes[1] = volume;
        else
            assert(false);

        volume->addFace(this);
    }
}

bool FaceConstraint::isBoundedBy(const VolumeConstraint* volume) const
{
    return _volumes[0] == volume || _volumes[1] == volume;
}

const AbstractConstraint* FaceConstraint::split(const AbstractConstraint* c) const
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

        if(isBoundedBy(v))
            return this;

        return SPLIT_VOLUME;
    }

    return SPLIT_VOLUME;
}

const AbstractConstraint* FaceConstraint::merge(const AbstractConstraint* c) const
{
    if(c->dimension() == 3)
    {
        return this;
    }
    else if(c->dimension() == 2)
    {
        const FaceConstraint* s =
            static_cast<const FaceConstraint*>(c);

        for(const EdgeConstraint* edge : _edges)
            if(edge->isBoundedBy(s))
                return edge;

        return PREVENT_MERGE;
    }
    else if(c->dimension() == 1)
    {
        const EdgeConstraint* e =
            static_cast<const EdgeConstraint*>(c);

        if(isBoundedBy(e))
            return e;

        for(const VertexConstraint* vertex : _vertices)
            if(vertex->isBoundedBy(e))
                return vertex;

        return PREVENT_MERGE;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        if(isBoundedBy(v))
            return v;

        return PREVENT_MERGE;
    }

    return PREVENT_MERGE;
}


PlaneConstraint::PlaneConstraint(int id, const glm::dvec3 &p, const glm::dvec3 &n) :
    FaceConstraint(id),
    _p(p), _n(glm::normalize(n))
{

}

glm::dvec3 PlaneConstraint::operator()(const glm::dvec3& pos) const
{
    glm::dvec3 dist = pos - _p;
    double l = glm::dot(dist, _n);
    return pos - _n * l;
}
