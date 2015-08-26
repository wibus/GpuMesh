#include "QualityLaplaceSmoother.h"

#include "../SmoothingHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


QualityLaplaceSmoother::QualityLaplaceSmoother() :
    AbstractVertexWiseSmoother({":/shaders/compute/Smoothing/QualityLaplace.glsl"})
{

}

QualityLaplaceSmoother::~QualityLaplaceSmoother()
{

}

const uint PROPOSITION_COUNT = 4;


void QualityLaplaceSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{
    std::vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    for(uint vId = first; vId < last; ++vId)
    {
        if(!SmoothingHelper::isSmoothable(mesh, vId))
            continue;


        // Compute patch center
        glm::dvec3 patchCenter = SmoothingHelper::computePatchCenter(mesh, vId);
        glm::dvec3& pos = verts[vId].p;
        glm::dvec3 centerDist = patchCenter - pos;


        // Define propositions for new vertex's position
        glm::dvec3 propositions[PROPOSITION_COUNT] = {
            pos,
            patchCenter - centerDist * _moveFactor,
            patchCenter,
            patchCenter + centerDist * _moveFactor,
        };

        const MeshTopo& topo = topos[vId];
        if(topo.isBoundary)
        {
            for(uint p=1; p < PROPOSITION_COUNT; ++p)
                propositions[p] = (*topo.snapToBoundary)(propositions[p]);
        }


        // Choose best position based on quality geometric mean
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
    }
}
