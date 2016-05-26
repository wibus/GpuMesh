#include "SphereBoundary.h"


SphereBoundary::SphereBoundary()
{

}

SphereBoundary::~SphereBoundary()
{

}

SphereBoundary::Surface::Surface() :
    SurfaceConstraint(1)
{

}

glm::dvec3 SphereBoundary::Surface::operator()(const glm::dvec3 &pos) const
{
    return glm::normalize(pos);
}
