#include "LocalOptimisationSmoother.h"

#include "OptimizationHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


LocalOptimisationSmoother::LocalOptimisationSmoother() :
    AbstractSmoother({OptimizationHelper::shaderName(),
                      ":/shaders/compute/Smoothing/LocalOptimisation.glsl"})
{

}

LocalOptimisationSmoother::~LocalOptimisationSmoother()
{

}


const uint PROPOSITION_COUNT = 4;

void LocalOptimisationSmoother::smoothVertices(
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

    for(int v = first; v < last; ++v)
    {
        const MeshTopo& topo = mesh.topo[v];
        if(topo.isFixed)
            continue;

        const vector<MeshNeigElem>& neighborElems = topo.neighborElems;
        if(neighborElems.empty())
            continue;


        // Compute local element size
        double localSize =
                OptimizationHelper::findLocalElementSize(
                    v, verts, topo.neighborVerts);


        // Define propositions to compute quality derivative
        glm::dvec3& pos = verts[v].p;
        double hStep = localSize / 10.0;
        glm::dvec3 propositions[PROPOSITION_COUNT] = {
            pos,
            pos + glm::dvec3(hStep, 0.0, 0.0),
            pos + glm::dvec3(0.0, hStep, 0.0),
            pos + glm::dvec3(0.0, 0.0, hStep),
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


        // Compute displacement
        glm::dvec3 gradQ = glm::dvec3(patchQualities[1] - patchQualities[0],
                                      patchQualities[2] - patchQualities[0],
                                      patchQualities[3] - patchQualities[0]) / hStep;
        double lambda = hStep / glm::length(gradQ);


        // Define proposed optimums
        propositions[0] = pos + gradQ * (lambda * (1.0 - _moveFactor));
        propositions[1] = pos + gradQ * (lambda * (1.0 - _moveFactor / 2.0));
        propositions[2] = pos + gradQ * (lambda * (1.0 + _moveFactor / 2.0));
        propositions[3] = pos + gradQ * (lambda * (1.0 + _moveFactor));

        if(topo.isBoundary)
            for(uint p=0; p < PROPOSITION_COUNT; ++p)
                propositions[p] = (*topo.snapToBoundary)(propositions[p]);

        patchQualities[0] = 1.0;
        patchQualities[1] = 1.0;
        patchQualities[2] = 1.0;
        patchQualities[3] = 1.0;


        // Compute proposed optimums patch quality
        OptimizationHelper::computePropositionPatchQualities(
                    mesh, v, topo,
                    neighborElems,
                    tets, pris, hexs,
                    evaluator,
                    propositions,
                    patchQualities,
                    PROPOSITION_COUNT);


        // Find best proposed optimum based on patch quality
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
