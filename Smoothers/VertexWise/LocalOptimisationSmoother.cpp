#include "LocalOptimisationSmoother.h"

#include <limits>

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


LocalOptimisationSmoother::LocalOptimisationSmoother() :
    AbstractVertexWiseSmoother(
        {":/shaders/compute/Smoothing/VertexWise/LocalOptimisation.glsl"}),
    _securityCycleCount(5),
    _localSizeToNodeShift(1.0 / 25.0)
{

}

LocalOptimisationSmoother::~LocalOptimisationSmoother()
{

}

void LocalOptimisationSmoother::setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program)
{
    AbstractVertexWiseSmoother::setVertexProgramUniforms(mesh, program);
    program.setInt("SecurityCycleCount", _securityCycleCount);
    program.setFloat("LocalSizeToNodeShift", _localSizeToNodeShift);
}

void LocalOptimisationSmoother::printSmoothingParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printSmoothingParameters(mesh, plot);
    plot.addSmoothingProperty("Method Name", "Local Optimization");
    plot.addSmoothingProperty("Local Size to Node Shift", to_string(_localSizeToNodeShift));
    plot.addSmoothingProperty("Security Cycle Count", to_string(_securityCycleCount));
}

void LocalOptimisationSmoother::smoothVertices(
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


        // Compute local element size
        double localSize =
                SmoothingHelper::computeLocalElementSize(
                    mesh, vId);

        // Initialize node shift distance
        double nodeShift = localSize * _localSizeToNodeShift;
        double originalNodeShift = nodeShift;

        for(int c=0; c < _securityCycleCount; ++c)
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
            if(topo.isBoundary)
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
                        SmoothingHelper::computePatchQuality(
                            mesh, evaluator, vId);
            }
            pos = originalPos;

            glm::dvec3 gradQ = glm::dvec3(
                sampleQualities[1] - sampleQualities[0],
                sampleQualities[3] - sampleQualities[2],
                sampleQualities[5] - sampleQualities[4]);
            double gradQNorm = glm::length(gradQ);

            if(gradQNorm == 0)
                break;


            const uint PROPOSITION_COUNT = 7;
            const double OFFSETS[PROPOSITION_COUNT] = {
                -0.25,
                 0.00,
                 0.25,
                 0.50,
                 0.75,
                 1.00,
                 1.25,
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
            };

            if(topo.isBoundary)
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
                        SmoothingHelper::computePatchQuality(
                            mesh, evaluator, vId);

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
