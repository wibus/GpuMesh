#include "ShellBoundary.h"


const double ShellBoundary::IN_RADIUS = 0.5;
const double ShellBoundary::OUT_RADIUS = 1.0;


ShellBoundary::ShellBoundary() :
    AbstractBoundary("Shell")
{
    volume()->addFace(&_inFace);
    volume()->addFace(&_outFace);
}

ShellBoundary::~ShellBoundary()
{

}
bool ShellBoundary::unitTest() const
{
    // Volume-Face
    assert(split(volume(), inFace())  == volume());
    assert(split(volume(), outFace()) == volume());
    assert(merge(volume(), inFace())  == inFace());
    assert(merge(volume(), outFace()) == outFace());

    // Face-Face
    assert(split(inFace(), outFace()) == volume());
    assert(merge(inFace(), outFace()) == INVALID_OPERATION);

    return true;
}


ShellBoundary::InFace::InFace() :
    FaceConstraint(2)
{

}

glm::dvec3 ShellBoundary::InFace::operator()(const glm::dvec3 &pos) const
{
    return glm::normalize(pos) * IN_RADIUS;
}


ShellBoundary::OutFace::OutFace() :
    FaceConstraint(1)
{

}

glm::dvec3 ShellBoundary::OutFace::operator()(const glm::dvec3 &pos) const
{
    return glm::normalize(pos);
}
