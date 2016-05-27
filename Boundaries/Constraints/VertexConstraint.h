#ifndef GPUMESH_VERTEX_CONSTRAINT
#define GPUMESH_VERTEX_CONSTRAINT

#include <vector>

#include "AbstractConstraint.h"


class VertexConstraint : public AbstractConstraint
{
public:
    VertexConstraint(int id, const glm::dvec3 position);

    void addEdge(const EdgeConstraint* edge);
    bool isBoundedBy(const EdgeConstraint* edge) const;

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override;

protected:
    virtual const AbstractConstraint* split(const AbstractConstraint* c) const override;
    virtual const AbstractConstraint* merge(const AbstractConstraint* c) const override;

private:
    glm::dvec3 _pos;
    std::vector<const EdgeConstraint*> _edges;
};


#endif // GPUMESH_VERTEX_CONSTRAINT
