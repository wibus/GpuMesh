#include "SpringLaplaceSmoother.h"

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;

const double SLMoveCoeff = 0.35;

// CUDA Drivers
void installCudaSpringLaplaceSmoother(float moveCoeff);
void installCudaSpringLaplaceSmoother()
{
    installCudaSpringLaplaceSmoother(SLMoveCoeff);
}
void smoothCudaVertices(const NodeGroups::GpuDispatch& dispatch);


SpringLaplaceSmoother::SpringLaplaceSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/SpringLaplace.glsl"},
        installCudaSpringLaplaceSmoother,
        smoothCudaVertices)
{

}

SpringLaplaceSmoother::~SpringLaplaceSmoother()
{

}

void SpringLaplaceSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
    program.setFloat("MoveCoeff", SLMoveCoeff);
}

void SpringLaplaceSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationImpl& plotImpl) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plotImpl);
    plotImpl.addSmoothingProperty("Method Name", "Spring Laplace");
    plotImpl.addSmoothingProperty("Move Factor", to_string(SLMoveCoeff));
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


        glm::dvec3 patchCenter =
            crew.measurer().computeVertexEquilibrium(
                mesh, crew.sampler(), vId);

        glm::dvec3& pos = verts[vId].p;
        pos = glm::mix(pos, patchCenter, SLMoveCoeff);

        const MeshTopo& topo = topos[vId];
        if(topo.snapToBoundary->isConstrained())
        {
            pos = (*topo.snapToBoundary)(pos);
        }
    }
}
