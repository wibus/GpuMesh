#include "SpringLaplaceSmoother.h"

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


SpringLaplaceSmoother::SpringLaplaceSmoother() :
    AbstractVertexWiseSmoother(
        {":/shaders/compute/Smoothing/VertexWise/SpringLaplace.glsl"})
{

}

SpringLaplaceSmoother::~SpringLaplaceSmoother()
{

}

void SpringLaplaceSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        const std::vector<uint>& vIds)
{
    std::vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    size_t vIdCount = vIds.size();
    for(int v = 0; v < vIdCount; ++v)
    {
        uint vId = vIds[v];

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

void SpringLaplaceSmoother::printSmoothingParameters(
        const Mesh& mesh,
        const AbstractEvaluator& evaluator,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printSmoothingParameters(mesh, evaluator, plot);
    plot.addSmoothingProperty("Method Name", "Spring Laplace");
    plot.addSmoothingProperty("Move Factor", to_string(_moveFactor));
}
