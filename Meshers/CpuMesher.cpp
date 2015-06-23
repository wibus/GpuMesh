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
        for(int e=0; e < MeshTet::EDGE_COUNT; ++e)
        {
            addEdge(_mesh.tetra[i].v[MeshTet::edges[e][0]],
                    _mesh.tetra[i].v[MeshTet::edges[e][1]]);
        }
    }

    int prismCount = _mesh.prism.size();
    for(int i=0; i < prismCount; ++i)
    {
        for(int e=0; e < MeshPri::EDGE_COUNT; ++e)
        {
            addEdge(_mesh.prism[i].v[MeshPri::edges[e][0]],
                    _mesh.prism[i].v[MeshPri::edges[e][1]]);
        }
    }

    int hexCount = _mesh.hexa.size();
    for(int i=0; i < hexCount; ++i)
    {
        for(int e=0; e < MeshHex::EDGE_COUNT; ++e)
        {
            addEdge(_mesh.hexa[i].v[MeshHex::edges[e][0]],
                    _mesh.hexa[i].v[MeshHex::edges[e][1]]);
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

void CpuMesher::addEdge(int firstVert, int secondVert)
{
    vector<int>& neighbors = _adjacency[firstVert].neighbors;
    int neighborCount = neighbors.size();
    for(int n=0; n < neighborCount; ++n)
    {
        if(secondVert == neighbors[n])
            return;
    }

    // This really is a new edge
    _adjacency[firstVert].neighbors.push_back(secondVert);
    _adjacency[secondVert].neighbors.push_back(firstVert);
}

void CpuMesher::smoothMesh()
{
    if(!_locationsComputed)
    {
        computeVertexLocations();
    }

    const double MOVE_FACTOR = 1.0;
    const double SMOOTH_AMELIORATION_THRESHOLD = 0.001;
    double dQuality = 1.0;
    int smoothPass = 0;


    double lastQualityMean, lastQualityVar;
    _mesh.compileElementQuality(
                lastQualityMean,
                lastQualityVar);

    cout << "Input mesh quality mean: " << lastQualityMean << endl;
    cout << "Input mesh quality std dev: " << lastQualityVar << endl;

    while(dQuality > SMOOTH_AMELIORATION_THRESHOLD)
    {
        int vertCount = _mesh.vert.size();
        for(int v = 0; v < vertCount; ++v)
        {
            glm::dvec3& vertPos = _mesh.vert[v].p;
            if(_mesh.vertProperties[v].isFixed)
                continue;

            const vector<int>& neighbors =
                    _adjacency[v].neighbors;

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

                const double alpha = 1.0 - MOVE_FACTOR;
                vertPos.x = alpha * vertPos.x + MOVE_FACTOR * barycenter.x;
                vertPos.y = alpha * vertPos.y + MOVE_FACTOR * barycenter.y;
                vertPos.z = alpha * vertPos.z + MOVE_FACTOR * barycenter.z;

                if(_mesh.vertProperties[v].isBoundary)
                {
                    vertPos = _mesh.vertProperties[v].boundaryCallback(vertPos);
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
