#include "CpuParametricMesher.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <iostream>

using namespace std;


const double PIPE_RADIUS = 0.3;
const glm::dvec3 EXT_NORMAL(-1, 0, 0);
const glm::dvec3 EXT_CENTER(-1, 0.5, 0.0);

class PipeSurfaceBoundary : public MeshBound
{
public:
    PipeSurfaceBoundary() :
        MeshBound(1)
    {

    }

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override
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
        glm::dvec3 extProj = glm::normalize(dist) * PIPE_RADIUS;
        return center + extProj;
    }
};

class PipeExtFaceBoundary : public MeshBound
{
public:
    PipeExtFaceBoundary() :
        MeshBound(2)
    {
    }

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override
    {
        glm::dvec3 center = EXT_CENTER;
        if(pos.y < 0.0) center.y = -center.y;

        double offset = glm::dot(pos - center, EXT_NORMAL);
        return pos - EXT_NORMAL * offset;
    }
};

class PipeExtEdgeBoundary : public MeshBound
{
public:
    PipeExtEdgeBoundary() :
        MeshBound(3)
    {
    }

    virtual glm::dvec3 operator()(const glm::dvec3& pos) const override
    {
        glm::dvec3 center = EXT_CENTER;
        if(pos.y < 0.0) center.y = -center.y;

        glm::dvec3 dist = pos - center;
        double offset = glm::dot(dist, EXT_NORMAL);
        glm::dvec3 extProj = dist - EXT_NORMAL * offset;
        return center + glm::normalize(extProj) * PIPE_RADIUS;
    }
};


CpuParametricMesher::CpuParametricMesher(unsigned int vertCount) :
    AbstractMesher(vertCount),
    _pipeSurface(new PipeSurfaceBoundary()),
    _pipeExtFace(new PipeExtFaceBoundary()),
    _pipeExtEdge(new PipeExtEdgeBoundary())
{

}

CpuParametricMesher::~CpuParametricMesher()
{

}

void CpuParametricMesher::triangulateDomain(Mesh& mesh)
{
    // Give proportianl dimensions
    int layerCount = 6;
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


    genStraightPipe(mesh,
                    glm::dvec3(-1.0, -0.5,  0),
                    glm::dvec3( 0.5, -0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    PIPE_RADIUS,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    true,
                    false);

    genArcPipe(mesh,
               glm::dvec3( 0.5,  0,    0),
               glm::dvec3( 0,    0,    1),
               glm::dvec3( 0,   -1.0,  0),
               glm::dvec3( 0,    0,    1),
               glm::pi<double>(),
               0.5,
               PIPE_RADIUS,
               arcPipeStackCount,
               sliceCount,
               layerCount,
               false,
               false);

    genStraightPipe(mesh,
                    glm::dvec3( 0.5,  0.5,  0),
                    glm::dvec3(-1.0,  0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    PIPE_RADIUS,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    false,
                    true);

    meshPipe(mesh, totalStackCount, sliceCount, layerCount);


    cout << "Elements / Vertices = " <<
            mesh.elemCount() << " / " << mesh.vertCount() << " = " <<
            mesh.elemCount()  / (double) mesh.vertCount() << endl;
}

void CpuParametricMesher::genStraightPipe(
        Mesh& mesh,
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
                    mesh,
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
        Mesh& mesh,
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
                    mesh,
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
        Mesh& mesh,
        const glm::dvec3& center,
        const glm::dvec4& upBase,
        const glm::dvec3& frontU,
        const glm::dmat4& dSlice,
        double dRadius,
        int sliceCount,
        int layerCount,
        bool isBoundary)
{
    MeshTopo extTopo(isBoundary ? *_pipeExtEdge : *_pipeSurface);
    MeshTopo intTopo(isBoundary ? *_pipeExtFace : MeshTopo::NO_BOUNDARY);

    mesh.vert.push_back(center);
    mesh.topo.push_back(intTopo);

    // Gen new stack vertices
    glm::dvec4 arm = upBase;
    for(int j=0; j<sliceCount; ++j, arm = dSlice * arm)
    {
        double radius = dRadius;
        for(int i=0; i<layerCount; ++i, radius += dRadius)
        {
            glm::dvec3 pos = center + glm::dvec3(arm) * radius;
            mesh.vert.push_back(pos);

            mesh.topo.push_back(
                (i == layerCount-1 ? extTopo : intTopo));
        }
    }
}

void CpuParametricMesher::meshPipe(
        Mesh& mesh,
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
            mesh.prism.push_back(
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

                mesh.hexa.push_back(
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
