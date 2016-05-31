#include "SpringLaplaceSmoother.h"

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;


// CUDA Drivers
void installCudaSpringLaplaceSmoother();


SpringLaplaceSmoother::SpringLaplaceSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/SpringLaplace.glsl"},
        installCudaSpringLaplaceSmoother)
{

}

SpringLaplaceSmoother::~SpringLaplaceSmoother()
{

}

void SpringLaplaceSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plot);
    plot.addSmoothingProperty("Method Name", "Spring Laplace");
    plot.addSmoothingProperty("Move Factor", to_string(_moveCoeff));
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
                mesh, crew.sampler(), vId);

        glm::dvec3& pos = verts[vId].p;
        pos = glm::mix(pos, patchCenter, _moveCoeff);

        const MeshTopo& topo = topos[vId];
        if(topo.snapToBoundary->isConstrained())
        {
            pos = (*topo.snapToBoundary)(pos);
        }
    }
}
