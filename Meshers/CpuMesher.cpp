#include "CpuMesher.h"

#include <iostream>

using namespace std;


struct Adjacency
{
    std::vector<int> neighbors;
};


CpuMesher::CpuMesher(Mesh& mesh, unsigned int vertCount) :
    AbstractMesher(mesh, vertCount),
    _locationsComputed(false)
{

}

CpuMesher::~CpuMesher()
{

}

void CpuMesher::computeVertexLocations()
{
    printStep(_stepId, "Computing vertex locations");

    int vertCount = _mesh.vert.size();

    _adjacency.clear();
    _adjacency.resize(vertCount);
    _adjacency.shrink_to_fit();

    int tetCount = _mesh.tetra.size();
    for(int i=0; i < tetCount; ++i)
    {
        const glm::ivec4& tet = _mesh.tetra[i];

        int verts[][2] = {
            {tet[0], tet[1]},
            {tet[0], tet[2]},
            {tet[0], tet[3]},
            {tet[1], tet[2]},
            {tet[1], tet[3]},
            {tet[2], tet[3]},
        };

        for(int e=0; e<6; ++e)
        {
            bool isPresent = false;
            int firstVert = verts[e][0];
            int secondVert = verts[e][1];
            vector<int>& neighbors = _adjacency[firstVert].neighbors;
            int neighborCount = neighbors.size();
            for(int n=0; n < neighborCount; ++n)
            {
                if(secondVert == neighbors[n])
                {
                    isPresent = true;
                    break;
                }
            }

            if(!isPresent)
            {
                _adjacency[firstVert].neighbors.push_back(secondVert);
                _adjacency[secondVert].neighbors.push_back(firstVert);
            }
        }
    }

    for(int i=0; i< vertCount; ++i)
    {
        _adjacency[i].neighbors.shrink_to_fit();
    }


    _locationsComputed = true;
}

void CpuMesher::clearVertexLocations()
{
    printStep(_stepId, "Clearing vertex locations");

    _adjacency.clear();
    _locationsComputed = false;
}

void CpuMesher::smoothMesh()
{
    if(!_locationsComputed)
    {
        computeVertexLocations();
    }

    const double MOVE_FACTOR = 0.5;
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
        for(int v = 0; v < vertCount; ++v)
        {
            glm::dvec4& vertPos = _mesh.vert[v];
            if(vertPos.w != 0.0)
                continue;

            double weightSum = 0.0;
            glm::dvec3 barycenter;

            for(auto& n : _adjacency[v].neighbors)
            {
                glm::dvec3 neighborPos(_mesh.vert[n]);
                glm::dvec3 dist = glm::dvec3(vertPos) - neighborPos;
                double weight = glm::log(glm::dot(dist, dist) + 1);

                barycenter = (barycenter * weightSum + neighborPos * weight)
                              / (weightSum + weight);
                weightSum += weight;
            }

            vertPos.x = glm::mix(vertPos.x, barycenter.x, MOVE_FACTOR);
            vertPos.y = glm::mix(vertPos.y, barycenter.y, MOVE_FACTOR);
            vertPos.z = glm::mix(vertPos.z, barycenter.z, MOVE_FACTOR);
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
