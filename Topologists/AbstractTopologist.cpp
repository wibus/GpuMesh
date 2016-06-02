#include "AbstractTopologist.h"


AbstractTopologist::AbstractTopologist() :
    _isEnabled(false),
    _frequency(1),
    _topoPassCount(3),
    _minEdgeLength(0.5),
    _maxEdgeLength(1.5)
{

}

AbstractTopologist::~AbstractTopologist()
{

}

void AbstractTopologist::setEnabled(bool enabled)
{
    _isEnabled = enabled;
}

void AbstractTopologist::setFrequency(int frequency)
{
    _frequency = frequency;
}

void AbstractTopologist::setTopoPassCount(int count)
{
    _topoPassCount = count;
}
