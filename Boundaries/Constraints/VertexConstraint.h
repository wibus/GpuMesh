#ifndef GPUMESH_VERTEX_CONSTRAINT
#define GPUMESH_VERTEX_CONSTRAINT

#include <vector>

#include "AbstractConstraint.h"


class VertexConstraint : public AbstractConstraint
{
public:
    VertexConstraint(int id, const glm::dvec3 position);

    void addEdge(EdgeConstraint* edge);
    bool isBoundedBy(const EdgeConstraint* edge) const;

    void addFace(FaceConstraint* face);
    bool isBoundedBy(const FaceConstraint* face) const;

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override;

    const glm::dvec3& position() const;

protected:
    virtual const AbstractConstraint* split(const AbstractConstraint* c) const override;
    virtual const AbstractConstraint* merge(const AbstractConstraint* c) const override;

private:
    glm::dvec3 _pos;
    std::vector<EdgeConstraint*> _edges;
    std::vector<FaceConstraint*> _faces;
};



// IMPLEMENTATION //
inline const glm::dvec3& VertexConstraint::position() const
{
    return _pos;
}


#endif // GPUMESH_VERTEX_CONSTRAINT
