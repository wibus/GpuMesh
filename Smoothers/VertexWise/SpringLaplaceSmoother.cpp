#include "SpringLaplaceSmoother.h"

#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;


SpringLaplaceSmoother::SpringLaplaceSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/SpringLaplace.glsl"})
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
        const MeshCrew& crew,
        const std::vector<uint>& vIds)
{
    std::vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    size_t vIdCount = vIds.size();
    for(int v = 0; v < vIdCount; ++v)
    {
        uint vId = vIds[v];

        if(!isSmoothable(mesh, vId))
            continue;

        glm::dvec3 patchCenter =
            crew.measurer().computeVertexEquilibrium(
                mesh, crew.discretizer(), vId);

        glm::dvec3& pos = verts[vId].p;
        pos = glm::mix(pos, patchCenter, _moveFactor);

        const MeshTopo& topo = topos[vId];
        if(topo.isBoundary)
        {
            pos = (*topo.snapToBoundary)(pos);
        }
    }
}
