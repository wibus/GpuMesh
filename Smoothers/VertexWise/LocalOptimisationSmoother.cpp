#include "LocalOptimisationSmoother.h"

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


LocalOptimisationSmoother::LocalOptimisationSmoother() :
    AbstractVertexWiseSmoother(DISPATCH_MODE_SCATTER,
        {":/shaders/compute/Smoothing/VertexWise/LocalOptimisation.glsl"})
{

}

LocalOptimisationSmoother::~LocalOptimisationSmoother()
{

}

void LocalOptimisationSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{
    std::vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;


    for(int vId = first; vId < last; ++vId)
    {
        if(!SmoothingHelper::isSmoothable(mesh, vId))
            continue;


        // Compute local element size
        double localSize =
                SmoothingHelper::computeLocalElementSize(
                    mesh, vId);

        // Initialize node shift distance
        double nodeShift = localSize / 25.0;
        double originalNodeShift = nodeShift;

        bool done = false;
        while(!done)
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
            double lambda = nodeShift / gradQNorm;
            double offsets[PROPOSITION_COUNT] = {
                -0.25,
                 0.00,
                 0.25,
                 0.50,
                 0.75,
                 1.00,
                 1.25,
            };

            glm::dvec3 propositions[PROPOSITION_COUNT] = {
                pos + gradQ * (lambda * offsets[0]),
                pos + gradQ * (lambda * offsets[1]),
                pos + gradQ * (lambda * offsets[2]),
                pos + gradQ * (lambda * offsets[3]),
                pos + gradQ * (lambda * offsets[4]),
                pos + gradQ * (lambda * offsets[5]),
                pos + gradQ * (lambda * offsets[6]),
            };

            if(topo.isBoundary)
            {
                for(uint p=0; p < PROPOSITION_COUNT; ++p)
                    propositions[p] = (*topo.snapToBoundary)(propositions[p]);
            }

            uint bestProposition = 0;
            double bestQualityMean = 0.0;
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
            nodeShift *= glm::abs(offsets[bestProposition]);
            if(nodeShift < originalNodeShift / 10.0)
                done = true;
        }
    }
}
