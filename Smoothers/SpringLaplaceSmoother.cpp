#include "SpringLaplaceSmoother.h"

#include "Evaluators/AbstractEvaluator.h"
#include "OptimizationHelper.h"

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
    for(int vId = first; vId < last; ++vId)
    {
        const MeshTopo& topo = mesh.topo[vId];
        if(topo.isFixed)
            continue;

        size_t neigElemCount = topo.neighborElems.size();
        if(neigElemCount == 0)
            continue;

        glm::dvec3 patchCenter =
            OptimizationHelper::computePatchCenter(
                mesh, vId);

        glm::dvec3& pos = mesh.vert[vId].p;
        pos = glm::mix(pos, patchCenter, _moveFactor);

        if(topo.isBoundary)
            pos = (*topo.snapToBoundary)(pos);
    }
}
