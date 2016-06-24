#include "AbstractTopologist.h"


AbstractTopologist::AbstractTopologist() :
    _minEdgeLength(0.5),
    _maxEdgeLength(1.5)
{

}

AbstractTopologist::~AbstractTopologist()
{

}

void AbstractTopologist::setMinEdgeLength(double length)
{
    _minEdgeLength = length;
}

void AbstractTopologist::setMaxEdgeLength(double length)
{
    _maxEdgeLength = length;
}
