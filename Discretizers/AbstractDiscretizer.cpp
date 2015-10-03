#include "AbstractDiscretizer.h"

#include "DataStructures/Mesh.h"


AbstractDiscretizer::AbstractDiscretizer()
{

}

AbstractDiscretizer::~AbstractDiscretizer()
{

}

double AbstractDiscretizer::vertValue(const Mesh& mesh, uint vId) const
{
    const glm::dvec3& vp = mesh.verts[vId].p;
    return (1.0 + sin((vp.x + vp.y) / (0.1 + vp.x*vp.x))) / 2.1 + 0.01;
}

void AbstractDiscretizer::boundingBox(
        const Mesh& mesh,
        glm::dvec3& minBounds,
        glm::dvec3& maxBounds) const
{
    minBounds = glm::dvec3(INFINITY);
    maxBounds = glm::dvec3(-INFINITY);
    size_t vertCount = mesh.verts.size();
    for(size_t v=0; v < vertCount; ++v)
    {
        const glm::dvec3& vertPos = mesh.verts[v].p;
        minBounds = glm::min(minBounds, vertPos);
        maxBounds = glm::max(maxBounds, vertPos);
    }
}
