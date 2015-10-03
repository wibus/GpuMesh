#include "AbstractDiscretizer.h"

#include "DataStructures/Mesh.h"


AbstractDiscretizer::AbstractDiscretizer()
{

}

AbstractDiscretizer::~AbstractDiscretizer()
{

}

double AbstractDiscretizer::vertValue(const Mesh& mesh, uint vId)
{
    const glm::dvec3& vp = mesh.verts[vId].p;
    return (1.0 + sin((vp.x + vp.y) / (0.1 + vp.x*vp.x))) / 2.1 + 0.01;
}
