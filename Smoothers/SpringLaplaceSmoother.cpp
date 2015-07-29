#include "SpringLaplaceSmoother.h"

#include <iostream>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;


SpringLaplaceSmoother::SpringLaplaceSmoother() :
    AbstractSmoother(":/shaders/compute/Smoothing/SpringLaplace.glsl")
{

}

SpringLaplaceSmoother::~SpringLaplaceSmoother()
{

}

void SpringLaplaceSmoother::smoothMeshCpp(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    _smoothPassId = 0;
    while(evaluateCpuMeshQuality(mesh, evaluator))
    {

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
                double weightSum = 0.0;
                glm::dvec3 patchCenter(0.0);

                int neigVertCount = neighborVerts.size();
                for(int i=0; i < neigVertCount; ++i)
                {
                    const glm::dvec3& npos = mesh.vert[neighborVerts[i]].p;

                    glm::dvec3 dist = npos - pos;
                    double weight = dot(dist, dist) + 0.0001;

                    patchCenter += npos * weight;
                    weightSum += weight;
                }

                patchCenter /= weightSum;
                pos += _moveFactor * (patchCenter - pos);

                if(topo.isBoundary)
                    pos = topo.snapToBoundary(pos);
            }
        }
    }

    mesh.updateGpuVertices();
}
