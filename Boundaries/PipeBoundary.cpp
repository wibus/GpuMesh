#include "PipeBoundary.h"


void installCudaPipeBoundary();


const double PipeBoundary::PIPE_RADIUS = 0.3;
const glm::dvec3 PipeBoundary::EXT_FACE_NORMAL(-1, 0, 0);
const glm::dvec3 PipeBoundary::EXT_YNEG_CENTER(-1, -0.5, 0.0);
const glm::dvec3 PipeBoundary::EXT_YPOS_CENTER(-1,  0.5, 0.0);


PipeBoundary::PipeBoundary() :
    AbstractBoundary("Pipe",
        ":/glsl/compute/Boundary/Pipe.glsl",
        installCudaPipeBoundary)
{
    volume()->addFace(&_cylinderFace);
    volume()->addFace(&_yNegDiskFace);
    volume()->addFace(&_yPosDiskFace);

    _cylinderFace.addEdge(&_yNegCircleEdge);
    _cylinderFace.addEdge(&_yPosCircleEdge);

    _yNegDiskFace.addEdge(&_yNegCircleEdge);
    _yPosDiskFace.addEdge(&_yPosCircleEdge);
}

PipeBoundary::~PipeBoundary()
{
    assert(split(volume(), cylinderFace()) == volume());
}

bool PipeBoundary::unitTest() const
{

    return true;
}


PipeBoundary::CylinderFace::CylinderFace() :
    FaceConstraint(1)
{

}

glm::dvec3 PipeBoundary::CylinderFace::operator()(const glm::dvec3 &pos) const
{
    glm::dvec3 center;

    if(pos.x < 0.5) // Straights
    {
        center = glm::dvec3(pos.x, (pos.y < 0.0 ? -0.5 : 0.5), 0.0);
    }
    else // Arc
    {
        center = pos - glm::dvec3(0.5, 0.0, pos.z);
        center = glm::normalize(center) * 0.5;
        center = glm::dvec3(0.5, 0, 0) + center;
    }

    glm::dvec3 dist = pos - center;
    glm::dvec3 extProj = glm::normalize(dist) * PIPE_RADIUS;
    return center + extProj;
}


PipeBoundary::YNegDiskFace::YNegDiskFace() :
    FaceConstraint(2)
{

}

glm::dvec3 PipeBoundary::YNegDiskFace::operator()(const glm::dvec3 &pos) const
{
    double offset = glm::dot(pos - EXT_YNEG_CENTER, EXT_FACE_NORMAL);
    return pos - EXT_FACE_NORMAL * offset;
}


PipeBoundary::YPosDiskFace::YPosDiskFace() :
    FaceConstraint(3)
{

}

glm::dvec3 PipeBoundary::YPosDiskFace::operator()(const glm::dvec3 &pos) const
{
    double offset = glm::dot(pos - EXT_YPOS_CENTER, EXT_FACE_NORMAL);
    return pos - EXT_FACE_NORMAL * offset;
}


PipeBoundary::YNegCircleEdge::YNegCircleEdge() :
    EdgeConstraint(4)
{

}

glm::dvec3 PipeBoundary::YNegCircleEdge::operator()(const glm::dvec3 &pos) const
{
    glm::dvec3 dist = pos - EXT_YNEG_CENTER;
    double offset = glm::dot(dist, EXT_FACE_NORMAL);
    glm::dvec3 extProj = dist - EXT_FACE_NORMAL * offset;
    return EXT_YNEG_CENTER + glm::normalize(extProj) * PIPE_RADIUS;
}


PipeBoundary::YPosCircleEdge::YPosCircleEdge() :
    EdgeConstraint(5)
{

}

glm::dvec3 PipeBoundary::YPosCircleEdge::operator()(const glm::dvec3 &pos) const
{
    glm::dvec3 dist = pos - EXT_YPOS_CENTER;
    double offset = glm::dot(dist, EXT_FACE_NORMAL);
    glm::dvec3 extProj = dist - EXT_FACE_NORMAL * offset;
    return EXT_YPOS_CENTER + glm::normalize(extProj) * PIPE_RADIUS;
}

