#ifndef GPUMESH_ABSTRACT_CONSTRAINT
#define GPUMESH_ABSTRACT_CONSTRAINT

#include <GLM/glm.hpp>


class AbstractBoundary;

// Basic Types of Constraint
class VertexConstraint;
class EdgeConstraint;
class SurfaceConstraint;
class VolumeConstraint;


class AbstractConstraint
{
protected:
    AbstractConstraint(int id, int dimension);

public:
    virtual ~AbstractConstraint();

    int id() const;
    int dimension() const;
    bool operator == (const AbstractConstraint& c) const;

    bool isFixed() const;
    bool isConstrained() const;

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const = 0;

    static const AbstractConstraint* SPLIT_VOLUME;
    static const AbstractConstraint* MERGE_PREVENT;

protected:
    friend class AbstractBoundary;
    virtual const AbstractConstraint* split(const AbstractConstraint* c) const = 0;
    virtual const AbstractConstraint* merge(const AbstractConstraint* c) const = 0;
private:
    int _id;
    int _dimension;
};



// IMPLEMENTATION //
inline int AbstractConstraint::id() const
{
    return _id;
}

inline int AbstractConstraint::dimension() const
{
    return _dimension;
}

inline bool AbstractConstraint::operator ==(const AbstractConstraint& c) const
{
    return _id == c.id();
}

inline bool AbstractConstraint::isFixed() const
{
    return _id < 0;
}

inline bool AbstractConstraint::isConstrained() const
{
    return _id != 0;
}


#endif // GPUMESH_ABSTRACT_CONSTRAINT
