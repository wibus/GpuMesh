#ifndef GPUMESH_CONSTRAINT
#define GPUMESH_CONSTRAINT

#include <GLM/glm.hpp>


// Basic Types of Constraint
class VertexConstraint;
class EdgeConstraint;
class SurfaceConstraint;
class VolumeConstraint;


class TopologyConstraint
{
protected:
    TopologyConstraint(int id, int dimension);

public:
    virtual ~TopologyConstraint();

    int id() const;
    int dimension() const;
    bool operator == (const TopologyConstraint& c) const;

    bool isFixed() const;
    bool isConstrained() const;

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const = 0;

    virtual const TopologyConstraint* split(const TopologyConstraint* c) const = 0;
    virtual const TopologyConstraint* merge(const TopologyConstraint* c) const = 0;

private:
    int _id;
    int _dimension;
};



// IMPLEMENTATION //
inline int TopologyConstraint::id() const
{
    return _id;
}

inline int TopologyConstraint::dimension() const
{
    return _dimension;
}

inline bool TopologyConstraint::operator ==(const TopologyConstraint& c) const
{
    return _id == c.id();
}

inline bool TopologyConstraint::isFixed() const
{
    return _id < 0;
}

inline bool TopologyConstraint::isConstrained() const
{
    return _id != 0;
}


#endif // GPUMESH_CONSTRAINT
