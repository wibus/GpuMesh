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

void QualityLaplaceSmoother::smoothCpuMesh(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    _smoothPassId = 0;
    while(evaluateCpuMeshQuality(mesh, evaluator))
    {
        size_t propositionCounts[] = {0, 0, 0, 0};

        int vertCount = mesh.vert.size();
        for(int v = 0; v < vertCount; ++v)
        {
            glm::dvec3& pos = mesh.vert[v].p;
            const MeshTopo& topo = mesh.topo[v];
            if(topo.isFixed)
                continue;

            const vector<MeshNeigVert>& neighborVerts = topo.neighborVerts;
            if(!neighborVerts.empty())
            {
                // Compute patch center
                glm::dvec3 patchCenter(0.0);
                int neigVertCount = neighborVerts.size();
                for(int i=0; i < neigVertCount; ++i)
                {
                    patchCenter += glm::dvec3(mesh.vert[neighborVerts[i]]);
                }
                patchCenter /= (double) neigVertCount;
                glm::dvec3 centerDist = patchCenter - pos;


                // Define propositions for new vertex's position
                const int PROPOSITION_COUNT = 4;
                const double AUX_DISTANCE = 0.2;
                glm::dvec3 propositions[PROPOSITION_COUNT] = {
                    pos,
                    patchCenter - AUX_DISTANCE * centerDist,
                    patchCenter,
                    patchCenter + AUX_DISTANCE * centerDist,
                };

                if(topo.isBoundary)
                    for(int p=1; p < PROPOSITION_COUNT; ++p)
                        propositions[p] = topo.boundaryCallback(propositions[p]);


                // Choose best position based on quality geometric mean
                int bestProposition = 0;
                double bestQualityMean = 0.0;
                size_t neighborElemCount = topo.neighborElems.size();
                for(int p=0; p < PROPOSITION_COUNT; ++p)
                {
                    double qualityGeometricMean = 1.0;
                    for(int n=0; n < neighborElemCount; ++n)
                    {
                        // Since 'pos' is a reference on vertex's position
                        // modifing its value here should be seen by the evaluator
                        pos = propositions[p];

                        const MeshNeigElem& neighborElem = topo.neighborElems[n];
                        if(neighborElem.type == MeshTet::ELEMENT_TYPE)
                        {
                            qualityGeometricMean *= evaluator.tetrahedronQuality(
                                mesh, mesh.tetra[neighborElem.id]);
                        }
                        else if(neighborElem.type == MeshPri::ELEMENT_TYPE)
                        {
                            qualityGeometricMean *= evaluator.prismQuality(
                                mesh, mesh.prism[neighborElem.id]);
                        }
                        else if(neighborElem.type == MeshHex::ELEMENT_TYPE)
                        {
                            qualityGeometricMean *= evaluator.hexahedronQuality(
                                mesh, mesh.hexa[neighborElem.id]);
                        }
                    }

                    qualityGeometricMean = glm::pow(
                        qualityGeometricMean,
                        1.0 / neighborElemCount);

                    if(qualityGeometricMean > bestQualityMean)
                    {
                        bestQualityMean = qualityGeometricMean;
                        bestProposition = p;
                    }
                }


                // Update vertex's position
                pos = propositions[bestProposition];
                ++propositionCounts[bestProposition];
            }
        }

        cout << "Propositions counts"
             << ": " << propositionCounts[0]
             << ", " << propositionCounts[1]
             << ", " << propositionCounts[2]
             << ", " << propositionCounts[3] << endl;
    }


    mesh.updateGpuVertices();
}
