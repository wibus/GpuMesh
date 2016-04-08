#include "AbstractTopologist.h"


AbstractTopologist::AbstractTopologist() :
    _isEnabled(false),
    _frequency(1)
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
