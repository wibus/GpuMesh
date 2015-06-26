#include "CpuLaplacianSmoother.h"

#include <iostream>

using namespace std;


CpuLaplacianSmoother::CpuLaplacianSmoother(
        Mesh &mesh,
        double moveFactor,
        double gainThreshold) :
    AbstractSmoother(mesh, moveFactor, gainThreshold)
{

}

CpuLaplacianSmoother::~CpuLaplacianSmoother()
{

}

void CpuLaplacianSmoother::smoothMesh()
{
    evaluateInitialMeshQuality();

    while(evaluateIterationMeshQuality())
    {

        int vertCount = _mesh.vert.size();
        for(int v = 0; v < vertCount; ++v)
        {
            glm::dvec3& pos = _mesh.vert[v].p;
            const MeshTopo& topo = _mesh.topo[v];
            if(topo.isFixed)
                continue;

            const vector<int>& neighbors = topo.neighbors;
            if(!neighbors.empty())
            {
                double weightSum = 0.0;
                glm::dvec3 barycenter;

                int neighborCount = neighbors.size();
                for(int i=0; i<neighborCount; ++i)
                {
                    int n = neighbors[i];
                    glm::dvec3 neighborPos(_mesh.vert[n]);
                    double weight = glm::length(pos - neighborPos) + 0.0001;
                    double alpha = weight / (weightSum + weight);

                    barycenter = glm::mix(barycenter, neighborPos, alpha);
                    weightSum += weight;
                }

                const double alpha = 1.0 - _moveFactor;
                pos.x = alpha * pos.x + _moveFactor * barycenter.x;
                pos.y = alpha * pos.y + _moveFactor * barycenter.y;
                pos.z = alpha * pos.z + _moveFactor * barycenter.z;

                if(topo.isBoundary)
                {
                    pos = topo.boundaryCallback(pos);
                }
            }
        }
    }

    cout << "#Smoothing finished" << endl << endl;
}
