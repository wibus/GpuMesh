#include "AbstractConstraint.h"


const AbstractConstraint* AbstractConstraint::SPLIT_VOLUME = nullptr;
const AbstractConstraint* AbstractConstraint::MERGE_PREVENT = nullptr;

AbstractConstraint::AbstractConstraint(int id, int dimension) :
    _id(id),
    _dimension(dimension)
{

}

AbstractConstraint::~AbstractConstraint()
{

}
