#ifndef GPUMESH_EDGE_CONSTRAINT
#define GPUMESH_EDGE_CONSTRAINT

#include "AbstractConstraint.h"


class EdgeConstraint : public AbstractConstraint
{
protected:
    EdgeConstraint(int id);

public:
    virtual const AbstractConstraint* subconstraint(int id) const override;

    void addVertex(VertexConstraint* vertex);
    bool isBoundedBy(const VertexConstraint* vertex) const;

    void addFace(FaceConstraint* face);
    bool isBoundedBy(const FaceConstraint* face) const;

protected:
    virtual const AbstractConstraint* split(const AbstractConstraint* c) const override;
    virtual const AbstractConstraint* merge(const AbstractConstraint* c) const override;

private:
    VertexConstraint* _vertices[2];
    FaceConstraint* _faces[2];
};


class SegmentConstraint : public EdgeConstraint
{
public:
    SegmentConstraint(int id, const glm::dvec3& a, const glm::dvec3& b);
    virtual glm::dvec3 operator ()(const glm::dvec3& pos) const override;
    const glm::dvec3& direction() const;
private:
    glm::dvec3 _a, _b, _u;
    double _length;
};



// IMPLEMENTATION //
inline const glm::dvec3& SegmentConstraint::direction() const
{
    return _u;
}


#endif // GPUMESH_EDGE_CONSTRAINT
