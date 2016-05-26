#ifndef GPUMESH_VERTEX_CONSTRAINT
#define GPUMESH_VERTEX_CONSTRAINT

#include <vector>

#include "Constraint.h"


class VertexConstraint : public TopologyConstraint
{
public:
    VertexConstraint(int id, const glm::dvec3 position);

    void addEdge(const EdgeConstraint* edge);

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override;

    virtual const TopologyConstraint* split(const TopologyConstraint* c) const override;
    virtual const TopologyConstraint* merge(const TopologyConstraint* c) const override;

private:
    glm::dvec3 _pos;
    std::vector<const EdgeConstraint*> _edges;
};


#endif // GPUMESH_VERTEX_CONSTRAINT
