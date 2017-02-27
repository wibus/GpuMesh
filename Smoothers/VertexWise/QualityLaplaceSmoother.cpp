#include "QualityLaplaceSmoother.h"

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;


const uint PROPOSITION_COUNT = 4;
const double QLMoveCoeff = 0.35;


// CUDA Drivers
void installCudaQualityLaplaceSmoother(float moveCoeff);
void installCudaQualityLaplaceSmoother()
{
    installCudaQualityLaplaceSmoother(QLMoveCoeff);
}
void smoothCudaVertices(const NodeGroups::GpuDispatch& dispatch);


QualityLaplaceSmoother::QualityLaplaceSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/QualityLaplace.glsl"},
        installCudaQualityLaplaceSmoother,
        smoothCudaVertices)
{

}

QualityLaplaceSmoother::~QualityLaplaceSmoother()
{

}

void QualityLaplaceSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
    program.setFloat("MoveCoeff", QLMoveCoeff);
}

void QualityLaplaceSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationImpl& plotImpl) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plotImpl);
    plotImpl.addSmoothingProperty("Method Name", "Quality Laplace");
    plotImpl.addSmoothingProperty("Line Sample Count", to_string(PROPOSITION_COUNT));
    plotImpl.addSmoothingProperty("Line Gaps", to_string(QLMoveCoeff));
}

void QualityLaplaceSmoother::smoothVertices(
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


        // Compute patch center
        glm::dvec3 patchCenter =
            crew.measurer().computeVertexEquilibrium(
                mesh, crew.sampler(), vId);

        glm::dvec3& pos = verts[vId].p;
        glm::dvec3 centerDist = patchCenter - pos;


        // Define propositions for new vertex's position
        glm::dvec3 propositions[PROPOSITION_COUNT] = {
            pos,
            patchCenter - centerDist * QLMoveCoeff,
            patchCenter,
            patchCenter + centerDist * QLMoveCoeff,
        };

        const MeshTopo& topo = topos[vId];
        if(topo.snapToBoundary->isConstrained())
        {
            for(uint p=1; p < PROPOSITION_COUNT; ++p)
                propositions[p] = (*topo.snapToBoundary)(propositions[p]);
        }


        // Choose best position
        uint bestProposition = 0;
        double bestQualityMean = -numeric_limits<double>::infinity();
        for(uint p=0; p < PROPOSITION_COUNT; ++p)
        {
            // Since 'pos' is a reference on vertex's position
            // modifing its value here should be seen by the evaluator
            pos = propositions[p];

            // Compute patch quality
            double patchQuality =
                crew.evaluator().patchQuality(
                    mesh, crew.sampler(), crew.measurer(), vId);

            if(patchQuality > bestQualityMean)
            {
                bestQualityMean = patchQuality;
                bestProposition = p;
            }
        }


        // Update vertex's position
        pos = propositions[bestProposition];
    }
}
