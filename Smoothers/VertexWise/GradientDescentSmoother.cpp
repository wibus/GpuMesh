#include "GradientDescentSmoother.h"

#include <limits>

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;


// Parameters
const int GDSecurityCycleCount = 5;
const double GDLocalSizeToNodeShift = 1.0 / 75.0;


// CUDA Drivers
void installCudaGradientDescentSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift);
void installCudaGradientDescentSmoother()
{
    installCudaGradientDescentSmoother(
                GDSecurityCycleCount,
                GDLocalSizeToNodeShift);
}
void smoothCudaVertices(const NodeGroups::GpuDispatch& dispatch);


GradientDescentSmoother::GradientDescentSmoother(
        const std::vector<std::string>& smoothShaders,
        const installCudaFct& installCuda,
        const launchCudaKernelFct& launchCudaKernel) :
    AbstractVertexWiseSmoother(smoothShaders, installCuda, launchCudaKernel)
{

}

GradientDescentSmoother::GradientDescentSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/GradientDescent.glsl"},
        installCudaGradientDescentSmoother,
        smoothCudaVertices)
{

}

GradientDescentSmoother::~GradientDescentSmoother()
{

}

void GradientDescentSmoother::setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program)
{
    AbstractVertexWiseSmoother::setVertexProgramUniforms(mesh, program);
    program.setInt("SecurityCycleCount", GDSecurityCycleCount);
    program.setFloat("LocalSizeToNodeShift", GDLocalSizeToNodeShift);
}

void GradientDescentSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationImpl& plotImpl) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plotImpl);
    plotImpl.addSmoothingProperty("Method Name", "Local Optimization");
    plotImpl.addSmoothingProperty("Local Size to Node Shift", to_string(GDLocalSizeToNodeShift));
    plotImpl.addSmoothingProperty("Security Cycle Count", to_string(GDSecurityCycleCount));
}

void GradientDescentSmoother::smoothVertices(
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


        // Compute local element size
        double localSize =
            crew.measurer().computeLocalElementSize(mesh, vId);

        // Initialize node shift distance
        double nodeShift = localSize * GDLocalSizeToNodeShift;
        double originalNodeShift = nodeShift;

        for(int c=0; c < GDSecurityCycleCount; ++c)
        {
            // Define patch quality gradient samples
            glm::dvec3& pos = verts[vId].p;
            const uint GRADIENT_SAMPLE_COUNT = 6;
            double sampleQualities[GRADIENT_SAMPLE_COUNT] =
                    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
            glm::dvec3 gradSamples[GRADIENT_SAMPLE_COUNT] = {
                pos + glm::dvec3(-nodeShift, 0.0,   0.0),
                pos + glm::dvec3( nodeShift, 0.0,   0.0),
                pos + glm::dvec3( 0.0,  -nodeShift, 0.0),
                pos + glm::dvec3( 0.0,   nodeShift, 0.0),
                pos + glm::dvec3( 0.0,   0.0,  -nodeShift),
                pos + glm::dvec3( 0.0,   0.0,   nodeShift),
            };

            const MeshTopo& topo = topos[vId];
            if(topo.snapToBoundary->isConstrained())
            {
                for(uint p=0; p < GRADIENT_SAMPLE_COUNT; ++p)
                    gradSamples[p] = (*topo.snapToBoundary)(gradSamples[p]);
            }

            glm::dvec3 originalPos = pos;
            for(uint p=0; p < GRADIENT_SAMPLE_COUNT; ++p)
            {
                // Since 'pos' is a reference on vertex's position
                // modifing its value here should be seen by the evaluator
                pos = gradSamples[p];

                // Compute patch quality
                sampleQualities[p] =
                    crew.evaluator().patchQuality(
                        mesh, crew.sampler(), crew.measurer(), vId);
            }
            pos = originalPos;

            glm::dvec3 gradQ = glm::dvec3(
                sampleQualities[1] - sampleQualities[0],
                sampleQualities[3] - sampleQualities[2],
                sampleQualities[5] - sampleQualities[4]);
            double gradQNorm = glm::length(gradQ);

            if(gradQNorm == 0)
                break;


            const uint PROPOSITION_COUNT = 8;
            const double OFFSETS[PROPOSITION_COUNT] = {
                -0.25,
                 0.00,
                 0.10,
                 0.20,
                 0.40,
                 0.80,
                 1.20,
                 1.60
            };

            glm::dvec3 shift = gradQ * (nodeShift / gradQNorm);
            glm::dvec3 propositions[PROPOSITION_COUNT] = {
                pos + shift * OFFSETS[0],
                pos + shift * OFFSETS[1],
                pos + shift * OFFSETS[2],
                pos + shift * OFFSETS[3],
                pos + shift * OFFSETS[4],
                pos + shift * OFFSETS[5],
                pos + shift * OFFSETS[6],
                pos + shift * OFFSETS[7]
            };

            if(topo.snapToBoundary->isConstrained())
            {
                for(uint p=0; p < PROPOSITION_COUNT; ++p)
                    propositions[p] = (*topo.snapToBoundary)(propositions[p]);
            }

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

            // Scale node shift and stop if it is too small
            nodeShift *= glm::abs(OFFSETS[bestProposition]);
            if(nodeShift < originalNodeShift / 10.0)
                break;
        }
    }
}
