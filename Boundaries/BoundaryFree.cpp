#include "BoundaryFree.h"


BoundaryFree::BoundaryFree() :
    AbstractBoundary("Free")
{
}

BoundaryFree::~BoundaryFree()
{

}

bool BoundaryFree::unitTest() const
{
    // Volume-Volume
    assert(split(volume(), volume()) == volume());
    assert(merge(volume(), volume()) == volume());

    return true;
}
