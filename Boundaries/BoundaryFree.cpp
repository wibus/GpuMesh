#include "BoundaryFree.h"


void installCudaBoundaryFree();


BoundaryFree::BoundaryFree() :
    AbstractBoundary("Free",
        ":/glsl/compute/Boundary/Free.glsl",
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
