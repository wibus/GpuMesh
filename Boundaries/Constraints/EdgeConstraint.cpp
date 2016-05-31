#include "EdgeConstraint.h"

#include "VertexConstraint.h"
#include "FaceConstraint.h"


EdgeConstraint::EdgeConstraint(int id) :
    AbstractConstraint(id, 1),
    _vertices{nullptr, nullptr},
    _faces{nullptr, nullptr}
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

        if(_faces[0] != nullptr)
            _faces[0]->addVertex(vertex);
        if(_faces[1] != nullptr)
            _faces[1]->addVertex(vertex);

        vertex->addEdge(this);
    }
}

bool EdgeConstraint::isBoundedBy(const VertexConstraint* vertex) const
{
    return vertex == _vertices[0] || vertex == _vertices[1];
}

void EdgeConstraint::addFace(FaceConstraint* face)
{
    if(!isBoundedBy(face))
    {
        if(_faces[0] == nullptr)
            _faces[0] = face;
        else if(_faces[1] == nullptr)
            _faces[1] = face;
        else
            assert(false);

        if(_vertices[0] != nullptr)
            _vertices[0]->addFace(face);
        if(_vertices[1] != nullptr)
            _vertices[1]->addFace(face);

        face->addEdge(this);
    }
}

bool EdgeConstraint::isBoundedBy(const FaceConstraint* face) const
{
    return face == _faces[0] || face == _faces[1];
}

const AbstractConstraint* EdgeConstraint::split(const AbstractConstraint* c) const
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

        for(const FaceConstraint* face : _faces)
            if(face->isBoundedBy(e))
                return face;

        return SPLIT_VOLUME;
    }
    else if(c->dimension() == 0)
    {
        const VertexConstraint* v =
            static_cast<const VertexConstraint*>(c);

        if(isBoundedBy(v))
            return v;

        for(const FaceConstraint* face : _faces)
            if(face->isBoundedBy(v))
                return face;

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
        const FaceConstraint* s =
            static_cast<const FaceConstraint*>(c);

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
