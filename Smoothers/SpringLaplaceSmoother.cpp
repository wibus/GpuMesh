#include "SpringLaplaceSmoother.h"

#include "SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


SpringLaplaceSmoother::SpringLaplaceSmoother() :
    AbstractSmoother({":/shaders/compute/Smoothing/SpringLaplace.glsl"})
{

}

SpringLaplaceSmoother::~SpringLaplaceSmoother()
{

}

void SpringLaplaceSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{
    std::vector<MeshVert>& verts = mesh.vert;
    const vector<MeshTopo>& topos = mesh.topo;

    for(int vId = first; vId < last; ++vId)
    {
        if(!SmoothingHelper::isSmoothable(mesh, vId))
            continue;

        glm::dvec3 patchCenter =
            SmoothingHelper::computePatchCenter(
                mesh, vId);

        glm::dvec3& pos = verts[vId].p;
        pos = glm::mix(pos, patchCenter, _moveFactor);

        const MeshTopo& topo = topos[vId];
        if(topo.isBoundary)
        {
            pos = (*topo.snapToBoundary)(pos);
        }
    }
}
