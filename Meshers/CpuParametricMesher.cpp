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
    int arcPipeStackCount = 20;
    int straightPipeStackCount = 20;
    int totalStackCount = 2 * straightPipeStackCount + arcPipeStackCount;

    int sliceCount = 20;
    int layerCount = 16;
    double pipeRadius = 0.3;

    genStraightPipe(glm::dvec3(-1.0, -0.5,  0),
                    glm::dvec3( 0.5, -0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    pipeRadius,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    true);

    genArcPipe(glm::dvec3( 0.5,  0,    0),
               glm::dvec3( 0,    0,    1),
               glm::dvec3( 0,   -1.0,  0),
               glm::dvec3( 0,    0,    1),
               glm::pi<double>(),
               0.5,
               pipeRadius,
               arcPipeStackCount,
               sliceCount,
               layerCount,
               false);

    genStraightPipe(glm::dvec3( 0.5,  0.5,  0),
                    glm::dvec3(-1.0,  0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    pipeRadius,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    false);

    meshPipe(totalStackCount, sliceCount, layerCount);
}

void CpuParametricMesher::genStraightPipe(
        const glm::dvec3& begin,
        const glm::dvec3& end,
        const glm::dvec3& up,
        double pipeRadius,
        int stackCount,
        int sliceCount,
        int layerCount,
        bool first)
{
    glm::dvec3 front = end - begin;
    double pipeLength = glm::length(front);
    glm::dvec3 frontU = front / pipeLength;

    glm::dvec4 armBase(glm::normalize(
        glm::cross(front, glm::cross(up, front))), 0.0);


    glm::dvec3 dFront = front / (double) stackCount;
    double dRadius = pipeRadius / (double) layerCount;
    glm::dmat4 dSlice = glm::rotate(glm::dmat4(),
        2.0 * glm::pi<double>() / sliceCount, frontU);


    glm::dvec3 center = begin;
    if(!first) center += dFront;
    for(int k= (first ? 0 : 1) ; k<=stackCount; ++k, center += dFront)
    {
        insertStackVertices(
                    center,
                    armBase,
                    dSlice,
                    dRadius,
                    sliceCount,
                    layerCount);
    }
}

void CpuParametricMesher::genArcPipe(
            const glm::dvec3& arcCenter,
            const glm::dvec3& rotationAxis,
            const glm::dvec3& dirBegin,
            const glm::dvec3& upBegin,
            double arcAngle,
            double arcRadius,
            double pipeRadius,
            int stackCount,
            int sliceCount,
            int layerCount,
            bool first)
{
    glm::dmat4 dStack = glm::rotate(glm::dmat4(),
        arcAngle / stackCount, rotationAxis);
    double dRadius = pipeRadius / (double) layerCount;

    glm::dvec4 dir(dirBegin, 0.0);
    glm::dvec4 up(upBegin, 0.0);
    glm::dvec4 front(glm::normalize(glm::cross(upBegin, dirBegin)), 0.0);
    if(!first)
    {
        front = dStack * front;
        dir = dStack * dir;
        up = dStack * up;
    }
    for(int k= (first ? 0 : 1) ;
        k <= stackCount; ++k,
        front = dStack * front,
        dir = dStack * dir,
        up = dStack * up)
    {
        glm::dvec3 center = arcCenter + glm::dvec3(dir) * arcRadius;
        glm::dmat4 dSlice = glm::rotate(glm::dmat4(),
            2.0 * glm::pi<double>() / sliceCount, glm::dvec3(front));

        insertStackVertices(
                    center,
                    up,
                    dSlice,
                    dRadius,
                    sliceCount,
                    layerCount);
    }
}

void CpuParametricMesher::insertStackVertices(
        const glm::dvec3& center,
        const glm::dvec4& upBase,
        const glm::dmat4& dSlice,
        double dRadius,
        int sliceCount,
        int layerCount)
{
    _mesh.vert.push_back(center);

    // Gen new stack vertices
    glm::dvec4 arm = upBase;
    for(int j=0; j<sliceCount; ++j, arm = dSlice * arm)
    {
        double radius = dRadius;
        for(int i=0; i<layerCount; ++i, radius += dRadius)
        {
            glm::dvec3 pos = center + glm::dvec3(arm) * radius;
            _mesh.vert.push_back(MeshVert(pos, i == layerCount-1));
        }
    }
}

void CpuParametricMesher::meshPipe(
            int stackCount,
            int sliceCount,
            int layerCount)
{
    int sliceVertCount = layerCount;
    int stackVertCount = sliceCount * sliceVertCount + 1; // +1 for center

    for(int k=1; k<=stackCount; ++k)
    {
        int maxK = k * stackVertCount;
        int minK = (k-1) * stackVertCount;


        for(int j=1; j<=sliceCount; ++j)
        {
            // +1 for center vertex
            int maxJ = (j % sliceCount) * sliceVertCount + 1;
            int minJ = (j - 1) * sliceVertCount + 1;


            // Create penta center
            _mesh.penta.push_back(
                MeshPen(
                    minJ + minK,
                    minJ + maxK,
                    maxJ + minK,
                    maxJ + maxK,
                    minK,
                    maxK
            ));


            // Create hex layers
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
}
