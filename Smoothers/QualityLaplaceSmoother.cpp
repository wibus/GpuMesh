#include "QualityLaplaceSmoother.h"

#include <iostream>

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

        int vertCount = mesh.vert.size();
        for(int v = 0; v < vertCount; ++v)
        {
            glm::dvec3& pos = mesh.vert[v].p;
            const MeshTopo& topo = mesh.topo[v];
            if(topo.isFixed)
                continue;

            const vector<int>& neighbors = topo.neighbors;
            if(!neighbors.empty())
            {
                double weightSum = 0.0;
                glm::dvec3 barycenter(0.0);

                int neighborCount = neighbors.size();
                for(int i=0; i<neighborCount; ++i)
                {
                    glm::dvec3 npos(mesh.vert[neighbors[i]]);

                    glm::dvec3 dist = npos - pos;
                    double weight = glm::dot(dist, dist) + 0.0001;

                    barycenter += npos * weight;
                    weightSum += weight;
                }

                barycenter /= weightSum;
                pos = glm::mix(pos, barycenter, _moveFactor);

                if(topo.isBoundary)
                {
                    pos = topo.boundaryCallback(pos);
                }
            }
        }
    }

    mesh.updateGpuVertices();

    cout << "#Smoothing finished" << endl << endl;
}
