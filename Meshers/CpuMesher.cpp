#include "CpuMesher.h"

#include <iostream>

using namespace std;


CpuMesher::CpuMesher(Mesh& mesh, unsigned int vertCount) :
    AbstractMesher(mesh, vertCount)
{

}

CpuMesher::~CpuMesher()
{

}

void CpuMesher::smoothMesh()
{
    const glm::dvec3 MOVE_FACTOR(0.5);
    const double SMOOTH_AMELIORATION_THRESHOLD = 0.001;
    double dQuality = 1.0;
    int smoothPass = 0;


    double lastQualityMean, lastQualityVar;
    _mesh.compileTetrahedronQuality(
                lastQualityMean,
                lastQualityVar);

    cout << "Input mesh quality mean: " << lastQualityMean << endl;
    cout << "Input mesh quality std dev: " << lastQualityVar << endl;

    while(dQuality > SMOOTH_AMELIORATION_THRESHOLD)
    {
        int vertCount = _mesh.vertCount();
        int firstVert = _mesh.externalVertCount;
        for(int v = firstVert; v < vertCount; ++v)
        {
            if(_mesh.vert[v].isBoundary)
                continue;

            double weightSum = 0.0;
            glm::dvec3 barycenter;

            glm::dvec3& vertPos = _mesh.vert[v].p;
            for(auto& n : _mesh.neighbors[v])
            {
                const glm::dvec3& neighborPos = _mesh.vert[n].p;
                glm::dvec3 dist = vertPos - neighborPos;
                double weight = glm::log(glm::dot(dist, dist) + 1);

                barycenter = (barycenter * weightSum + neighborPos * weight)
                              / (weightSum + weight);
                weightSum += weight;
            }

            vertPos = glm::mix(vertPos, barycenter, MOVE_FACTOR);
        }

        double newQualityMean, newQualityVar;
        _mesh.compileTetrahedronQuality(
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
