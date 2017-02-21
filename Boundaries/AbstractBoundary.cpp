#include "AbstractBoundary.h"

#include "Constraints/VertexConstraint.h"
#include "Constraints/EdgeConstraint.h"
#include "Constraints/FaceConstraint.h"
#include "Constraints/VolumeConstraint.h"


const AbstractConstraint* AbstractBoundary::INVALID_OPERATION = nullptr;

AbstractBoundary::AbstractBoundary(const std::string& name) :
    _name(name)
{

}

AbstractBoundary::~AbstractBoundary()
{

}

const AbstractConstraint* AbstractBoundary::constraint(int id) const
{
    const AbstractConstraint* c = nullptr;
    c = volume()->subconstraint(id);
    if(c != nullptr)
        return c;

    return volume();
}

int AbstractBoundary::supportDimension(
    const AbstractConstraint* c1,
    const AbstractConstraint* c2) const
{
    return split(c1, c2)->dimension();
}

const AbstractConstraint* AbstractBoundary::split(
    const AbstractConstraint* c1,
    const AbstractConstraint* c2) const
{
    if(c1 == c2)
        return c1;

    const AbstractConstraint* constraint = c1->split(c2);

    if(constraint != AbstractConstraint::SPLIT_VOLUME)
        return constraint;
    else
        return volume();
}

const AbstractConstraint* AbstractBoundary::merge(
    const AbstractConstraint* c1,
    const AbstractConstraint* c2) const
{
    if(c1 == c2)
        return c1;

    const AbstractConstraint* constraint = c1->merge(c2);

    if(constraint == AbstractConstraint::PREVENT_MERGE)
        return INVALID_OPERATION;
    else
        return constraint;
}
