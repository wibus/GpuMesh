#include "QualityLaplaceSmoother.h"

#include "OptimizationHelper.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


QualityLaplaceSmoother::QualityLaplaceSmoother() :
    AbstractSmoother({":/shaders/compute/Smoothing/QualityLaplace.glsl"})
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

        size_t neigElemCount = topo.neighborElems.size();
        if(neigElemCount == 0)
            continue;


        // Compute patch center
        glm::dvec3 patchCenter =
                OptimizationHelper::findPatchCenter(
                    v, topo, verts,
                    tets, pris, hexs);

        glm::dvec3& pos = verts[v].p;
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

        // Choose best position based on quality geometric mean
        uint bestProposition = 0;
        double bestQualityMean = 0.0;
        for(uint p=0; p < PROPOSITION_COUNT; ++p)
        {
            // Since 'pos' is a reference on vertex's position
            // modifing its value here should be seen by the evaluator
            pos = propositions[p];

            double patchQuality = 1.0;
            for(size_t n=0; n < neigElemCount; ++n)
            {

                const MeshNeigElem& neigElem = topo.neighborElems[n];
                switch(neigElem.type)
                {
                case MeshTet::ELEMENT_TYPE:
                    OptimizationHelper::accumulatePatchQuality(
                        evaluator.tetQuality(mesh, tets[neigElem.id]),
                        patchQuality);
                    break;

                case MeshPri::ELEMENT_TYPE:
                    OptimizationHelper::accumulatePatchQuality(
                        evaluator.priQuality(mesh, pris[neigElem.id]),
                        patchQuality);
                    break;

                case MeshHex::ELEMENT_TYPE:
                    OptimizationHelper::accumulatePatchQuality(
                        evaluator.hexQuality(mesh, hexs[neigElem.id]),
                        patchQuality);
                    break;
                }

                if(patchQuality <= 0.0)
                {
                    break;
                }
            }

            OptimizationHelper::finalizePatchQuality(patchQuality);

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
