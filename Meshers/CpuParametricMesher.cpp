#include "CpuParametricMesher.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <iostream>

using namespace std;


class PipeSurfaceBoundary
{
public:
    PipeSurfaceBoundary(double radius) :
        radius(radius)
    {

    }

    glm::dvec3 operator()(const glm::dvec3& pos) const
    {
        glm::dvec3 center;

        if(pos.x < 0.5) // Straights
        {
            center = glm::dvec3(pos.x, (pos.y < 0.0 ? -0.5 : 0.5), 0.0);

        }
        else // Arc
        {
            center = pos - glm::dvec3(0.5, 0.0, pos.z);
            center = glm::normalize(center) * 0.5;
            center = glm::dvec3(0.5, 0, 0) + center;
        }

        glm::dvec3 dist = pos - center;
        glm::dvec3 extProj = glm::normalize(dist) * radius;
        return center + extProj;
    }

private:
    double radius;
};

class PipeExtremityFaceBoundary
{
public:
    PipeExtremityFaceBoundary(
            const glm::dvec3& center,
            const glm::dvec3& normal) :
        center(center),
        normal(normal)
    {
    }

    glm::dvec3 operator()(const glm::dvec3& pos) const
    {
        double offset = glm::dot(pos - center, normal);
        return pos - normal * offset;
    }

private:
    glm::dvec3 center;
    glm::dvec3 normal;
};

class PipeExtremityEdgeBoundary
{
public:
    PipeExtremityEdgeBoundary(
            const glm::dvec3& center,
            const glm::dvec3& normal,
            double radius) :
        center(center),
        normal(normal),
        radius(radius)
    {
    }

    glm::dvec3 operator()(const glm::dvec3& pos) const
    {
        glm::dvec3 dist = pos - center;
        double offset = glm::dot(dist, normal);
        glm::dvec3 extProj = dist - normal * offset;
        return center + glm::normalize(extProj) * radius;
    }

private:
    glm::dvec3 center;
    glm::dvec3 normal;
    double radius;
};


CpuParametricMesher::CpuParametricMesher(Mesh& mesh, unsigned int vertCount) :
    AbstractMesher(mesh, vertCount)
{

}

CpuParametricMesher::~CpuParametricMesher()
{

}

void CpuParametricMesher::triangulateDomain()
{
    double pipeRadius = 0.3;

    // Give proportianl dimensions
    int layerCount = 8;
    int sliceCount = 20;
    int arcPipeStackCount = 30;
    int straightPipeStackCount = 30;
    int totalStackCount = 2 * straightPipeStackCount + arcPipeStackCount;

    // Rescale dimension to fit vert count hint
    int vertCount = (layerCount * sliceCount + 1) * totalStackCount;
    double scaleFactor = glm::pow(_vertCount / (double) vertCount, 1/3.0);
    layerCount = glm::floor(layerCount * scaleFactor);
    sliceCount = glm::ceil(sliceCount * scaleFactor);
    arcPipeStackCount = glm::ceil(arcPipeStackCount * scaleFactor);
    straightPipeStackCount = glm::ceil(straightPipeStackCount * scaleFactor);
    totalStackCount = 2 * straightPipeStackCount + arcPipeStackCount;


    genStraightPipe(glm::dvec3(-1.0, -0.5,  0),
                    glm::dvec3( 0.5, -0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    pipeRadius,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    true,
                    false);

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
               false,
               false);

    genStraightPipe(glm::dvec3( 0.5,  0.5,  0),
                    glm::dvec3(-1.0,  0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    pipeRadius,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    false,
                    true);

    meshPipe(totalStackCount, sliceCount, layerCount);


    cout << "Elements / Vertices = " <<
            _mesh.elemCount() << " / " << _mesh.vertCount() << " = " <<
            _mesh.elemCount()  / (double) _mesh.vertCount() << endl;
}

void CpuParametricMesher::genStraightPipe(
        const glm::dvec3& begin,
        const glm::dvec3& end,
        const glm::dvec3& up,
        double pipeRadius,
        int stackCount,
        int sliceCount,
        int layerCount,
        bool first,
        bool last)
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
        bool isBoundary = (k == 0) || (last && k == stackCount);
        insertStackVertices(
                    center,
                    armBase,
                    frontU,
                    dSlice,
                    dRadius,
                    sliceCount,
                    layerCount,
                    isBoundary);
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
            bool first,
            bool last)
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
        glm::dvec3 frontU = glm::dvec3(front);
        glm::dvec3 center = arcCenter + glm::dvec3(dir) * arcRadius;
        glm::dmat4 dSlice = glm::rotate(glm::dmat4(),
            2.0 * glm::pi<double>() / sliceCount, frontU);

        bool isBoundary = (k == 0) || (last && k == stackCount);

        insertStackVertices(
                    center,
                    up,
                    frontU,
                    dSlice,
                    dRadius,
                    sliceCount,
                    layerCount,
                    isBoundary);
    }
}

void CpuParametricMesher::insertStackVertices(
        const glm::dvec3& center,
        const glm::dvec4& upBase,
        const glm::dvec3& frontU,
        const glm::dmat4& dSlice,
        double dRadius,
        int sliceCount,
        int layerCount,
        bool isBoundary)
{
    MeshTopo extTopo;
    MeshTopo intTopo;

    if(isBoundary)
    {
        extTopo.isBoundary = true;
        extTopo.boundaryCallback =
                PipeExtremityEdgeBoundary(center, frontU, layerCount * dRadius);

        intTopo.isBoundary = true;
        intTopo.boundaryCallback =
                PipeExtremityFaceBoundary(center, frontU);
    }
    else
    {
        extTopo.isBoundary = true;
        extTopo.boundaryCallback =
                PipeSurfaceBoundary(layerCount * dRadius);
    }

    _mesh.vert.push_back(center);
    _mesh.topo.push_back(intTopo);

    // Gen new stack vertices
    glm::dvec4 arm = upBase;
    for(int j=0; j<sliceCount; ++j, arm = dSlice * arm)
    {
        double radius = dRadius;
        for(int i=0; i<layerCount; ++i, radius += dRadius)
        {
            glm::dvec3 pos = center + glm::dvec3(arm) * radius;
            _mesh.vert.push_back(pos);

            _mesh.topo.push_back(
                (i == layerCount-1 ? extTopo : intTopo));
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
            _mesh.prism.push_back(
                MeshPri(
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
