#ifndef GPUMESH_SURFACE_CONSTRAINT
#define GPUMESH_SURFACE_CONSTRAINT

#include <vector>

#include "Constraint.h"


class SurfaceConstraint : public TopologyConstraint
{
protected:
    SurfaceConstraint(int id);

public:
    void addEdge(EdgeConstraint* edge);
    void addVolume(const VolumeConstraint* volume);
    bool isBoundedBy(const EdgeConstraint* edge) const;

    virtual const TopologyConstraint* split(const TopologyConstraint* c) const override;
    virtual const TopologyConstraint* merge(const TopologyConstraint* c) const override;

private:
    std::vector<const EdgeConstraint*> _edges;
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
