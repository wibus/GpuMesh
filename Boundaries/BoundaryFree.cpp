#include "BoundaryFree.h"


void installCudaBoundaryFree();


BoundaryFree::BoundaryFree() :
    AbstractBoundary("Free",
        installCudaBoundaryFree)
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
