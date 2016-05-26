#include "ShellBoundary.h"


ShellBoundary::ShellBoundary()
{

}

ShellBoundary::~ShellBoundary()
{

}


ShellBoundary::InSurface::InSurface() :
    SurfaceConstraint(2)
{

}

glm::dvec3 ShellBoundary::InSurface::operator()(const glm::dvec3 &pos) const
{
    return glm::normalize(pos) * 0.5;
}


ShellBoundary::OutSurface::OutSurface() :
    SurfaceConstraint(1)
{

}

glm::dvec3 ShellBoundary::OutSurface::operator()(const glm::dvec3 &pos) const
{
    return glm::normalize(pos);
}
