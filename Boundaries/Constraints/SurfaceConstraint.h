#ifndef GPUMESH_SURFACE_CONSTRAINT
#define GPUMESH_SURFACE_CONSTRAINT

#include <vector>

#include "AbstractConstraint.h"


class SurfaceConstraint : public AbstractConstraint
{
protected:
    SurfaceConstraint(int id);

public:
    void addVertex(VertexConstraint* vertex);
    bool isBoundedBy(const VertexConstraint* vertex) const;

    void addEdge(EdgeConstraint* edge);
    bool isBoundedBy(const EdgeConstraint* edge) const;

    void addVolume(VolumeConstraint* volume);
    bool isBoundedBy(const VolumeConstraint* volume) const;

protected:
    virtual const AbstractConstraint* split(const AbstractConstraint* c) const override;
    virtual const AbstractConstraint* merge(const AbstractConstraint* c) const override;

private:
    std::vector<VertexConstraint*> _vertices;
    std::vector<EdgeConstraint*> _edges;
    const VolumeConstraint* _volumes[2];
};


class PlaneConstraint : public SurfaceConstraint
{
public:
    PlaneConstraint(int id, const glm::dvec3& p, const glm::dvec3& n);
    virtual glm::dvec3 operator ()(const glm::dvec3& pos) const override;

private:
    glm::dvec3 _p, _n;
};


#endif // GPUMESH_SURFACE_CONSTRAINT
