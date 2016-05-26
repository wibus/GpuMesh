#ifndef GPUMESH_EDGE_CONSTRAINT
#define GPUMESH_EDGE_CONSTRAINT

#include "Constraint.h"


class EdgeConstraint : public TopologyConstraint
{
protected:
    EdgeConstraint(int id);

public:
    void addVertex(VertexConstraint* vertex);
    bool isBoundedBy(const VertexConstraint* v) const;

    void addSurface(const SurfaceConstraint* surface);
    const SurfaceConstraint* getSurface(const EdgeConstraint* edge);

    virtual const TopologyConstraint* split(const TopologyConstraint* c) const override;
    virtual const TopologyConstraint* merge(const TopologyConstraint* c) const override;

private:
    const VertexConstraint* _vertices[2];
    const SurfaceConstraint* _surfaces[2];
};


class SegmentConstraint : public EdgeConstraint
{
public:
    SegmentConstraint(int id, const glm::dvec3& a, const glm::dvec3& b);
    virtual glm::dvec3 operator ()(const glm::dvec3& pos) const override;
private:
    glm::dvec3 _a, _b, _u;
    double _length;
};


#endif // GPUMESH_EDGE_CONSTRAINT
