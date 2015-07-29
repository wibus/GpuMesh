#include "QualityLaplaceSmoother.h"

#include <iostream>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;


QualityLaplaceSmoother::QualityLaplaceSmoother() :
    AbstractSmoother(":/shaders/compute/Smoothing/QualityLaplace.glsl")
{

}

QualityLaplaceSmoother::~QualityLaplaceSmoother()
{

}

void QualityLaplaceSmoother::smoothMeshCpp(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    std::vector<MeshVert>& verts = mesh.vert;
    std::vector<MeshTet>& tets = mesh.tetra;
    std::vector<MeshPri>& pris = mesh.prism;
    std::vector<MeshHex>& hexs = mesh.hexa;


    _smoothPassId = 0;
    while(evaluateCpuMeshQuality(mesh, evaluator))
    {
        size_t vertCount = verts.size();
        for(size_t v = 0; v < vertCount; ++v)
        {
            const MeshTopo& topo = mesh.topo[v];
            const vector<MeshNeigElem>& neighborElems = topo.neighborElems;
            size_t neigElemCount = neighborElems.size();
            if(topo.isFixed || neigElemCount == 0)
                continue;

            // Compute patch center
            size_t totalVertCount = 0;
            glm::dvec3 patchCenter(0.0);
            for(size_t n=0; n < neigElemCount; ++n)
            {
                const MeshNeigElem& neigElem = neighborElems[n];
                switch(neigElem.type)
                {
                case MeshTet::ELEMENT_TYPE:
                    totalVertCount += MeshTet::VERTEX_COUNT - 1;
                    for(size_t i=0; i < MeshTet::VERTEX_COUNT; ++i)
                        patchCenter += verts[tets[neigElem.id].v[i]].p;
                    break;

                case MeshPri::ELEMENT_TYPE:
                    totalVertCount += MeshPri::VERTEX_COUNT - 1;
                    for(size_t i=0; i < MeshPri::VERTEX_COUNT; ++i)
                        patchCenter += verts[pris[neigElem.id].v[i]].p;
                    break;

                case MeshHex::ELEMENT_TYPE:
                    totalVertCount += MeshHex::VERTEX_COUNT - 1;
                    for(size_t i=0; i < MeshHex::VERTEX_COUNT; ++i)
                        patchCenter += verts[hexs[neigElem.id].v[i]].p;
                    break;
                }
            }

            glm::dvec3& pos = verts[v].p;
            patchCenter = (patchCenter - pos * double(neigElemCount))
                            / double(totalVertCount);
            glm::dvec3 centerDist = patchCenter - pos;


            // Define propositions for new vertex's position
            const uint PROPOSITION_COUNT = 4;
            glm::dvec3 propositions[PROPOSITION_COUNT] = {
                pos,
                patchCenter - centerDist * _moveFactor,
                patchCenter,
                patchCenter + centerDist * _moveFactor,
            };

            if(topo.isBoundary)
                for(uint p=1; p < PROPOSITION_COUNT; ++p)
                    propositions[p] = topo.snapToBoundary(propositions[p]);


            // Choose best position based on quality geometric mean
            uint bestProposition = 0;
            double bestQualityMean = 0.0;
            for(uint p=0; p < PROPOSITION_COUNT; ++p)
            {
                double qualityGeometricMean = 1.0;
                for(size_t n=0; n < neigElemCount; ++n)
                {
                    // Since 'pos' is a reference on vertex's position
                    // modifing its value here should be seen by the evaluator
                    pos = propositions[p];

                    const MeshNeigElem& neigElem = topo.neighborElems[n];
                    switch(neigElem.type)
                    {
                    case MeshTet::ELEMENT_TYPE:
                        qualityGeometricMean *= evaluator.tetQuality(
                            mesh, tets[neigElem.id]);
                        break;

                    case MeshPri::ELEMENT_TYPE:
                        qualityGeometricMean *= evaluator.priQuality(
                            mesh, pris[neigElem.id]);
                        break;

                    case MeshHex::ELEMENT_TYPE:
                        qualityGeometricMean *= evaluator.hexQuality(
                            mesh, hexs[neigElem.id]);
                        break;
                    }

                    if(qualityGeometricMean <= 0.0)
                    {
                        qualityGeometricMean = 0.0;
                        break;
                    }
                }

                if(qualityGeometricMean > bestQualityMean)
                {
                    bestQualityMean = qualityGeometricMean;
                    bestProposition = p;
                }
            }


            // Update vertex's position
            pos = propositions[bestProposition];
        }
    }


    mesh.updateGpuVertices();
}
