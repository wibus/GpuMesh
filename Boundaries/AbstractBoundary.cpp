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

    // nullptr constraint means that the
    // split vertex floats in the volume
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

    // Nullptr constraint means it can't be merged
    if(constraint == AbstractConstraint::MERGE_PREVENT)
        return INVALID_OPERATION;
    else
        return constraint;
}
