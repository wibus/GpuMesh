#include "AbstractSmoother.h"


AbstractSmoother::AbstractSmoother(
        Mesh &mesh,
        double moveFactor,
        double gainThreshold) :
    _mesh(mesh),
    _moveFactor(moveFactor),
    _gainThreshold(gainThreshold)
{
}

AbstractSmoother::~AbstractSmoother()
{

}
