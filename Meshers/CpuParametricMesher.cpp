#include "CpuParametricMesher.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <iostream>

using namespace std;


CpuParametricMesher::CpuParametricMesher(Mesh& mesh, unsigned int vertCount) :
    CpuMesher(mesh, vertCount)
{

}

CpuParametricMesher::~CpuParametricMesher()
{

}

void CpuParametricMesher::triangulateDomain()
{
    genStraightTube(glm::dvec3(1, 0, 0),
                    glm::dvec3(-1, 0, 0),
                    glm::dvec3(0, 0, 1),
                    0.3,
                    20,
                    10,
                    6);
}

void CpuParametricMesher::genStraightTube(
        const glm::dvec3& begin,
        const glm::dvec3& end,
        const glm::dvec3& up,
        double tubeRadius,
        int stackCount,
        int sliceCount,
        int layerCount)
{
    glm::dvec3 front = end - begin;
    double tubeLength = glm::length(front);
    glm::dvec3 frontU = front / tubeLength;

    glm::dvec3 dFront = front / (double) stackCount;
    double dRadius = tubeRadius / (double) layerCount;
    glm::dmat4 dRotMat = glm::rotate(glm::dmat4(),
        2.0 * glm::pi<double>() / sliceCount, frontU);


    int stackVertCount = sliceCount * layerCount + 1; // +1 for center

    vector<glm::dvec3> lastStack(stackVertCount);
    vector<glm::dvec3> currStack(stackVertCount);

    glm::dvec3 center = begin;
    for(int k=0; k<=stackCount; ++k, center += dFront)
    {
        currStack[0] = center;
        _mesh.vert.push_back(center);

        // Gen new stack vertices
        glm::dvec4 arm(up, 0.0);
        for(int j=0; j<sliceCount; ++j, arm = dRotMat * arm)
        {
            int jId = j * layerCount + 1;

            double radius = dRadius;
            for(int i=0; i<layerCount; ++i, radius += dRadius)
            {
                glm::dvec3 pos = center + glm::dvec3(arm) * radius;
                _mesh.vert.push_back(MeshVert(pos, i == layerCount-1));
                currStack[jId + i] = pos;
            }
        }

        if(k != 0)
        {
            int maxK = k * stackVertCount;
            int minK = (k-1) * stackVertCount;

            // Create penta center

            // Create hex layers
            for(int j=1; j<=sliceCount; ++j)
            {
                int maxJ = (j % sliceCount) * layerCount + 1;
                int minJ = (j - 1) * layerCount + 1;

                _mesh.penta.push_back(
                    MeshPen(
                        minJ + minK,
                        minJ + maxK,
                        maxJ + minK,
                        maxJ + maxK,
                        minK,
                        maxK
                ));


                for(int i=1; i<layerCount; ++i)
                {
                    int maxI = i;
                    int minI = i-1;

                    _mesh.hexa.push_back(
                        MeshHex(
                            minI + minJ + minK,
                            maxI + minJ + minK,
                            minI + maxJ + minK,
                            maxI + maxJ + minK,
                            minI + minJ + maxK,
                            maxI + minJ + maxK,
                            minI + maxJ + maxK,
                            maxI + maxJ + maxK
                    ));
                }
            }
        }

        lastStack.swap(currStack);
    }
}
