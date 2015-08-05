#include "QualityLaplaceSmoother.h"

#include "OptimizationHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


QualityLaplaceSmoother::QualityLaplaceSmoother() :
    AbstractSmoother({OptimizationHelper::shaderName(),
                      ":/shaders/compute/Smoothing/QualityLaplace.glsl"})
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
    std::vector<MeshVert>& verts = mesh.vert;
    std::vector<MeshTet>& tets = mesh.tetra;
    std::vector<MeshPri>& pris = mesh.prism;
    std::vector<MeshHex>& hexs = mesh.hexa;

    for(uint v = first; v < last; ++v)
    {
        const MeshTopo& topo = mesh.topo[v];
        if(topo.isFixed)
            continue;

        const vector<MeshNeigElem>& neighborElems = topo.neighborElems;
        if(neighborElems.empty())
            continue;

        // Compute patch center
        glm::dvec3& pos = verts[v].p;
        glm::dvec3 patchCenter =
                OptimizationHelper::findPatchCenter(
                    v, verts,
                    neighborElems,
                    tets, pris, hexs);
        glm::dvec3 centerDist = patchCenter - pos;


        // Define propositions for new vertex's position
        glm::dvec3 propositions[PROPOSITION_COUNT] = {
            pos,
            patchCenter - centerDist * _moveFactor,
            patchCenter,
            patchCenter + centerDist * _moveFactor,
        };

        if(topo.isBoundary)
            for(uint p=1; p < PROPOSITION_COUNT; ++p)
                propositions[p] = (*topo.snapToBoundary)(propositions[p]);

        double patchQualities[PROPOSITION_COUNT] = {1.0, 1.0, 1.0, 1.0};


        // Compute proposition's patch quality
        OptimizationHelper::computePropositionPatchQualities(
                    mesh, v, topo,
                    neighborElems,
                    tets, pris, hexs,
                    evaluator,
                    propositions,
                    patchQualities,
                    PROPOSITION_COUNT);

        // Find best proposition based on patch quality
        uint bestProposition = 0;
        double bestQualityResult = patchQualities[0];
        for(uint p=1; p < PROPOSITION_COUNT; ++p)
        {
            if(bestQualityResult < patchQualities[p])
            {
                bestProposition = p;
                bestQualityResult = patchQualities[p];
            }
        }

        // Update vertex's position
        pos = propositions[bestProposition];
    }
}
