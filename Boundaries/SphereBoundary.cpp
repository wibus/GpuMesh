#include "SphereBoundary.h"

#include <CellarWorkbench/Misc/Log.h>

using namespace cellar;


void installCudaSphereBoundary();


const double SphereBoundary::RADIUS = 1.0;


SphereBoundary::SphereBoundary() :
    AbstractBoundary("Sphere",
        ":/glsl/compute/Boundary/Sphere.glsl",
        installCudaSphereBoundary)
{
    volume()->addFace(&_face);
}

SphereBoundary::~SphereBoundary()
{

}

bool SphereBoundary::unitTest() const
{
    // Volume-Volume
    assert(split(volume(), volume()) == volume());
    assert(merge(volume(), volume()) == volume());

    // Volume-Face
    assert(split(volume(), face()) == volume());
    assert(merge(volume(), face()) == face());

    // Face-Face
    assert(split(face(), face()) == face());
    assert(merge(face(), face()) == face());

    return true;
}

SphereBoundary::Face::Face() :
    FaceConstraint(1)
{
}

glm::dvec3 SphereBoundary::Face::operator()(const glm::dvec3 &pos) const
{
    return glm::normalize(pos);
}
