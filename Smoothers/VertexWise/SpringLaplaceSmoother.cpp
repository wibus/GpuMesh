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

void SpringLaplaceSmoother::printSmoothingParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printSmoothingParameters(mesh, plot);
    plot.addSmoothingProperty("Method Name", "Spring Laplace");
    plot.addSmoothingProperty("Move Factor", to_string(_moveFactor));
}

void SpringLaplaceSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        const AbstractDiscretizer& discretizer,
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
                mesh, discretizer, vId);

        glm::dvec3& pos = verts[vId].p;
        pos = glm::mix(pos, patchCenter, _moveFactor);

        const MeshTopo& topo = topos[vId];
        if(topo.isBoundary)
        {
            pos = (*topo.snapToBoundary)(pos);
        }
    }
}
