#include "CpuLagrangianSmoother.h"

#include <iostream>

using namespace std;


CpuLangrangianSmoother::CpuLangrangianSmoother(
        Mesh &mesh,
        double moveFactor,
        double gainThreshold) :
    AbstractSmoother(mesh, moveFactor, gainThreshold)
{

}

CpuLangrangianSmoother::~CpuLangrangianSmoother()
{

}

void CpuLangrangianSmoother::smoothMesh()
{
    double dQuality = 1.0;
    int smoothPass = 0;


    double lastQualityMean, lastQualityVar;
    _mesh.compileElementQuality(
                lastQualityMean,
                lastQualityVar);

    cout << "Input mesh quality mean: " << lastQualityMean << endl;
    cout << "Input mesh quality std dev: " << lastQualityVar << endl;

    while(dQuality > _gainThreshold)
    {
        int vertCount = _mesh.vert.size();
        for(int v = 0; v < vertCount; ++v)
        {
            glm::dvec3& vertPos = _mesh.vert[v].p;
            const MeshVertProperties& vertProp = _mesh.vertProperties[v];
            if(vertProp.isFixed)
                continue;

            const vector<int>& neighbors = vertProp.neighbors;
            if(!neighbors.empty())
            {
                double weightSum = 0.0;
                glm::dvec3 barycenter;

                int neighborCount = neighbors.size();
                for(int i=0; i<neighborCount; ++i)
                {
                    int n = neighbors[i];
                    glm::dvec3 neighborPos(_mesh.vert[n]);
                    glm::dvec3 dist = glm::dvec3(vertPos) - neighborPos;
                    double weight = glm::dot(dist, dist) + 0.1;

                    barycenter = (barycenter * weightSum + neighborPos * weight)
                                  / (weightSum + weight);
                    weightSum += weight;
                }

                const double alpha = 1.0 - _moveFactor;
                vertPos.x = alpha * vertPos.x + _moveFactor * barycenter.x;
                vertPos.y = alpha * vertPos.y + _moveFactor * barycenter.y;
                vertPos.z = alpha * vertPos.z + _moveFactor * barycenter.z;

                if(vertProp.isBoundary)
                {
                    vertPos = vertProp.boundaryCallback(vertPos);
                }
            }
        }

        double newQualityMean, newQualityVar;
        _mesh.compileElementQuality(
                    newQualityMean,
                    newQualityVar);

        cout << "Smooth pass number " << smoothPass << endl;
        cout << "Mesh quality mean: " << newQualityMean << endl;
        cout << "Mesh quality std dev: " << newQualityVar << endl;

        dQuality = newQualityMean - lastQualityMean;
        lastQualityMean = newQualityMean;
        lastQualityVar = newQualityVar;
        ++smoothPass;
    }

    cout << "#Smoothing finished" << endl << endl;
}
