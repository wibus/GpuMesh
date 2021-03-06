#include "CpuParametricMesher.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Misc/Log.h>

#include "Boundaries/PipeBoundary.h"

using namespace std;
using namespace cellar;


CpuParametricMesher::CpuParametricMesher() :
    _pipeBoundary(new PipeBoundary())
{
    using namespace std::placeholders;
    _modelFuncs.setDefault("Pipe");
    _modelFuncs.setContent({
        {string("Pipe"),   ModelFunc(bind(&CpuParametricMesher::genPipe,   this, _1, _2))},
        {string("Bottle"), ModelFunc(bind(&CpuParametricMesher::genBottle, this, _1, _2))},
    });
}

CpuParametricMesher::~CpuParametricMesher()
{

}

void CpuParametricMesher::genPipe(Mesh& mesh, size_t vertexCount)
{
    // Give proportianl dimensions
    int layerCount = 6;
    int sliceCount = 20;
    int arcPipeStackCount = 30;
    int straightPipeStackCount = 30;
    int totalStackCount = 2 * straightPipeStackCount + arcPipeStackCount;

    // Rescale dimension to fit vert count hint
    int vertCount = (layerCount * sliceCount + 1) * totalStackCount;
    double scaleFactor = glm::pow(vertexCount / (double) vertCount, 1/3.0);
    layerCount = glm::floor(layerCount * scaleFactor);
    sliceCount = glm::ceil(sliceCount * scaleFactor);
    arcPipeStackCount = glm::ceil(arcPipeStackCount * scaleFactor);
    straightPipeStackCount = glm::ceil(straightPipeStackCount * scaleFactor);
    totalStackCount = 2 * straightPipeStackCount + arcPipeStackCount;


    insertStraightRingPipe(mesh,
                    glm::dvec3(-1.0, -0.5,  0),
                    glm::dvec3( 0.5, -0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    PipeBoundary::PIPE_RADIUS,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    true,
                    false);

    insertArcRingPipe(mesh,
               glm::dvec3( 0.5,  0,    0),
               glm::dvec3( 0,    0,    1),
               glm::dvec3( 0,   -1.0,  0),
               glm::dvec3( 0,    0,    1),
               glm::pi<double>(),
               0.5,
               PipeBoundary::PIPE_RADIUS,
               arcPipeStackCount,
               sliceCount,
               layerCount,
               false,
               false);

    insertStraightRingPipe(mesh,
                    glm::dvec3( 0.5,  0.5,  0),
                    glm::dvec3(-1.0,  0.5,  0),
                    glm::dvec3( 0,    0,    1),
                    PipeBoundary::PIPE_RADIUS,
                    straightPipeStackCount,
                    sliceCount,
                    layerCount,
                    false,
                    true);

    meshPipe(mesh, totalStackCount, sliceCount, layerCount);
    mesh.setBoundary(_pipeBoundary);
}

void CpuParametricMesher::genBottle(Mesh& mesh, size_t vertexCount)
{
    // Physic dimensions
    double bodyRadius = 0.15;
    double bodyThickness = 0.05;
    double bodyHeight = 0.65;

    double heelRadius = bodyRadius;
    double heelThickness = 0.05;

    double neckRadius = 0.04;
    double neckThickness = bodyThickness * (neckRadius / bodyRadius);
    double neckOutRadius = neckRadius + neckThickness;
    double neckLength = 0.35;

    double bottleBottomZ = -bodyHeight - bodyRadius + heelThickness;


    // Discrete dimensions
    int wallThickN = 5;

    int polySideN = 6;
    int heelRingN = 20;

    int bodySilceN = wallThickN * (bodyHeight / heelThickness);
    int bodyDivN = polySideN * heelRingN;

    int shoulderSliceN = 2 * wallThickN * (bodyRadius / heelThickness);

    int neckSilceN = wallThickN * neckLength / neckThickness;
    int neckDivN = polySideN * heelRingN;


    // Bottle heel
    glm::dmat3 heelRot = glm::dmat3(
        glm::rotate(
            glm::dmat4(),
            glm::pi<double>() * 2.0 / polySideN,
            glm::dvec3(0.0, 0.0, 1.0)));

    size_t heelBaseIdx = mesh.verts.size();
    for(int s=0; s <= wallThickN; ++s)
    {
        double heelZ = (s * heelThickness / wallThickN) + bottleBottomZ;

        // Center vertex
        glm::dvec3 ringCenter(0.0, 0.0, heelZ);
        mesh.verts.push_back(ringCenter);

        for(int r=1; r <= heelRingN; ++r)
        {
            double heelR = (r * heelRadius / heelRingN);
            glm::dvec3 heelArm(heelR, 0.0, 0.0);
            glm::dvec3 nextArm = heelRot * heelArm;

            for(int d=0; d < polySideN; ++d)
            {
                // First side vertex
                mesh.verts.push_back(heelArm + ringCenter);

                glm::dvec3 heelStep = (nextArm - heelArm) / double(r);
                for(int v=1; v < r; ++v)
                {
                    // additionnal side vertices
                    mesh.verts.push_back(
                        glm::normalize(heelArm + heelStep*double(v)) * heelR
                                + ringCenter);
                }

                heelArm = nextArm;
                nextArm = heelRot * nextArm;
            }
        }
    }

    size_t lastSliceBaseIdx = heelBaseIdx;
    size_t currSliceBaseIdx = lastSliceBaseIdx;
    size_t sliceVertexCount = 1 + polySideN*(heelRingN)*(heelRingN+1) / 2;
    for(int s=1; s <= wallThickN; ++s)
    {
        currSliceBaseIdx += sliceVertexCount;

        size_t lastRingBaseIdx = 0;
        size_t currRingBaseIdx = 1;
        for(int r=1; r <= heelRingN; ++r)
        {
            size_t lastRingVertCount = (r-1)*polySideN;
            size_t currRingVertCount = r*polySideN;
            currRingBaseIdx += lastRingVertCount;

            for(int d=0; d < polySideN; ++d)
            {
                size_t lastRingSideBaseIdx = d*(r-1);
                size_t currRingSideBaseIdx = d*r;
                size_t currRingSideNextIdx =
                        (currRingSideBaseIdx + 1)%currRingVertCount;

                mesh.pris.push_back(MeshPri(
                    lastSliceBaseIdx + lastRingBaseIdx + lastRingSideBaseIdx,
                    lastSliceBaseIdx + currRingBaseIdx + currRingSideBaseIdx,
                    lastSliceBaseIdx + currRingBaseIdx + currRingSideNextIdx,
                    currSliceBaseIdx + lastRingBaseIdx + lastRingSideBaseIdx,
                    currSliceBaseIdx + currRingBaseIdx + currRingSideBaseIdx,
                    currSliceBaseIdx + currRingBaseIdx + currRingSideNextIdx));

                size_t currRingSideStepIdx = currRingSideNextIdx;
                size_t lastRingSideStepIdx = lastRingSideBaseIdx;
                for(int v=1; v < r; ++v)
                {
                    currRingSideNextIdx = (currRingSideStepIdx + 1)%currRingVertCount;
                    size_t lastRingSideNextIdx = (lastRingSideStepIdx + 1)%lastRingVertCount;

                    mesh.pris.push_back(MeshPri(
                        lastSliceBaseIdx + lastRingBaseIdx + lastRingSideStepIdx,
                        lastSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        lastSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx,
                        currSliceBaseIdx + lastRingBaseIdx + lastRingSideStepIdx,
                        currSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        currSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx));

                    mesh.pris.push_back(MeshPri(
                        lastSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx,
                        lastSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        lastSliceBaseIdx + currRingBaseIdx + currRingSideNextIdx,
                        currSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx,
                        currSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        currSliceBaseIdx + currRingBaseIdx + currRingSideNextIdx));

                    ++currRingSideStepIdx;
                    ++lastRingSideStepIdx;
                }

            }

            lastRingBaseIdx = currRingBaseIdx;
        }

        lastSliceBaseIdx = currSliceBaseIdx;
    }


    // Bottle body
    glm::dmat3 bodyRot = glm::dmat3(
        glm::rotate(
            glm::dmat4(),
            glm::pi<double>() * 2.0 / bodyDivN,
            glm::dvec3(0.0, 0.0, 1.0)));

    size_t bodyBaseIdx = mesh.verts.size();
    for(int s=0; s <= bodySilceN; ++s)
    {
        double bodyZ = (s * bodyHeight / bodySilceN) + bottleBottomZ;
        glm::dvec3 ringCenter(0.0, 0.0, bodyZ);

        for(int r=1; r <= wallThickN; ++r)
        {
            double rProg = r / double(wallThickN);
            double bodyR = bodyRadius + bodyThickness * rProg;
            glm::dvec3 bodyArm(bodyR, 0.0, 0.0);

            for(int d=0; d < bodyDivN; ++ d)
            {
                mesh.verts.push_back(bodyArm + ringCenter);
                bodyArm = bodyRot * bodyArm;
            }
        }
    }


    // Bottle shoulder
    double shoulderRotNormalization = glm::pi<double>() * 0.5 /
        (bodyRadius*shoulderSliceN + ((neckOutRadius - bodyRadius) / (shoulderSliceN)) *
            (shoulderSliceN) * (shoulderSliceN+1) / 2.0);

    glm::dvec3 shoulder(1.0, 0.0, 0.0);
    glm::dvec3 shoulderCenter(neckOutRadius, 0.0, bottleBottomZ + bodyHeight);
    for(int s=1; s < shoulderSliceN; ++s)
    {
        double sProg = double(s) / shoulderSliceN;

        double radius = bodyRadius - shoulderCenter.x +
                        glm::mix(0.0, bodyThickness - neckThickness, sProg);

        double thickness = glm::mix(bodyThickness, neckThickness, sProg);

        double rotAngle = glm::mix(bodyRadius, neckOutRadius, sProg)
                            * shoulderRotNormalization;
        glm::dmat3 shoulderRot = glm::dmat3(
            glm::rotate(
                glm::dmat4(),
                rotAngle,
                glm::dvec3(0.0, -1.0, 0.0)));

        shoulder = shoulderRot * shoulder;

        for(int r=1; r <= wallThickN; ++r)
        {
            double rProg = double(r) / wallThickN;
            double bodyR = radius + thickness * rProg;
            glm::dvec3 shoulderArm = shoulder * bodyR + shoulderCenter;

            for(int d=0; d < bodyDivN; ++ d)
            {
                mesh.verts.push_back(shoulderArm);
                shoulderArm = bodyRot * shoulderArm;
            }
        }
    }

    size_t lastStackBaseIdx = bodyBaseIdx;
    size_t currStackBaseIdx = lastStackBaseIdx;
    size_t stackVertexCount = bodyDivN * wallThickN;
    for(int s=1; s < bodySilceN + shoulderSliceN; ++s)
    {
        currStackBaseIdx += stackVertexCount;

        if(s <= wallThickN)
        {
            size_t lastSliceBaseIdx = (s-1)*sliceVertexCount + (1+polySideN*(heelRingN-1)*(heelRingN)/2);
            size_t currSliceBaseIdx = lastSliceBaseIdx + sliceVertexCount;

            // Prism-Hexahedra connection
            for(int d=0; d < bodyDivN; ++d)
            {
                int nextD = (d+1)%bodyDivN;
                mesh.hexs.push_back(MeshHex(
                    currSliceBaseIdx + d,
                    lastSliceBaseIdx + d,
                    lastSliceBaseIdx + nextD,
                    currSliceBaseIdx + nextD,
                    currStackBaseIdx + d,
                    lastStackBaseIdx + d,
                    lastStackBaseIdx + nextD,
                    currStackBaseIdx + nextD));
            }
        }

        for(int r=1; r < wallThickN; ++r)
        {
            size_t lastRingIdx = (r-1) * bodyDivN;
            size_t currRingIdx = r * bodyDivN;

            for(int d=0; d < bodyDivN; ++d)
            {
                int nextD = (d+1)%bodyDivN;
                mesh.hexs.push_back(MeshHex(
                    currStackBaseIdx + lastRingIdx + d,
                    lastStackBaseIdx + lastRingIdx + d,
                    lastStackBaseIdx + lastRingIdx + nextD,
                    currStackBaseIdx + lastRingIdx + nextD,
                    currStackBaseIdx + currRingIdx + d,
                    lastStackBaseIdx + currRingIdx + d,
                    lastStackBaseIdx + currRingIdx + nextD,
                    currStackBaseIdx + currRingIdx + nextD));
            }
        }

        lastStackBaseIdx = currStackBaseIdx;
    }


    // Bottle neck
    glm::dmat3 neckRot = glm::dmat3(
        glm::rotate(
            glm::dmat4(),
            glm::pi<double>() * 2.0 / neckDivN,
            glm::dvec3(0.0, 0.0, 1.0)));

    size_t neckBaseIdx = mesh.verts.size();
    for(int s=1; s <= neckSilceN; ++s)
    {
        double neckZ = (s * neckLength / neckSilceN)
            + bottleBottomZ + bodyHeight +
            (bodyRadius + bodyThickness - (neckRadius + 2.0*neckThickness));
        glm::dvec3 ringCenter(0.0, 0.0, neckZ);

        for(int r=0; r <= wallThickN; ++r)
        {
            double rProg = double(r) / wallThickN;
            double neckR = neckRadius + rProg * neckThickness;
            glm::dvec3 neckArm(neckR, 0.0, 0.0);

            for(int d=0; d < neckDivN; ++ d)
            {
                mesh.verts.push_back(neckArm + ringCenter);
                neckArm = neckRot * neckArm;
            }
        }
    }

    lastStackBaseIdx = neckBaseIdx;
    currStackBaseIdx = lastStackBaseIdx;
    stackVertexCount = neckDivN * (wallThickN+1);
    for(int s=2; s <= neckSilceN; ++s)
    {
        currStackBaseIdx += stackVertexCount;

        if(s <= wallThickN)
        {
            size_t ultimateRingIdx = wallThickN * neckDivN;
            size_t lastRingIdx = neckBaseIdx - wallThickN * bodyDivN + (s-2) * bodyDivN;
            size_t currRingIdx = neckBaseIdx - wallThickN * bodyDivN + (s-1) * bodyDivN;

            for(int d=0; d < neckDivN; ++d)
            {
                int nextD = (d+1)%neckDivN;
                mesh.hexs.push_back(MeshHex(
                    currStackBaseIdx + ultimateRingIdx + d,
                    lastStackBaseIdx + ultimateRingIdx + d,
                    lastStackBaseIdx + ultimateRingIdx + nextD,
                    currStackBaseIdx + ultimateRingIdx + nextD,
                    currRingIdx + d,
                    lastRingIdx + d,
                    lastRingIdx + nextD,
                    currRingIdx + nextD));
            }
        }

        for(int r=1; r <= wallThickN; ++r)
        {
            size_t lastRingIdx = (r-1) * neckDivN;
            size_t currRingIdx = r * neckDivN;

            for(int d=0; d < neckDivN; ++d)
            {
                int nextD = (d+1)%neckDivN;
                mesh.hexs.push_back(MeshHex(
                    currStackBaseIdx + lastRingIdx + d,
                    lastStackBaseIdx + lastRingIdx + d,
                    lastStackBaseIdx + lastRingIdx + nextD,
                    currStackBaseIdx + lastRingIdx + nextD,
                    currStackBaseIdx + currRingIdx + d,
                    lastStackBaseIdx + currRingIdx + d,
                    lastStackBaseIdx + currRingIdx + nextD,
                    currStackBaseIdx + currRingIdx + nextD));
            }
        }

        lastStackBaseIdx = currStackBaseIdx;
    }
}

void CpuParametricMesher::insertStraightRingPipe(
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
        insertRingStackVertices(
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

void CpuParametricMesher::insertArcRingPipe(
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

        insertRingStackVertices(
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

void CpuParametricMesher::insertRingStackVertices(
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
    MeshTopo extTopo(!isBoundary ?
        static_cast<const AbstractConstraint*>(_pipeBoundary->cylinderFace()) :
        static_cast<const AbstractConstraint*>(center.y < 0.0 ? _pipeBoundary->yNegCircleEdge() :
                          _pipeBoundary->yPosCircleEdge()));
    MeshTopo intTopo(!isBoundary ?
        static_cast<const AbstractConstraint*>(MeshTopo::NO_BOUNDARY) :
        static_cast<const AbstractConstraint*>(center.y < 0.0 ? _pipeBoundary->yNegDiskFace() :
                          _pipeBoundary->yPosDiskFace()));

    mesh.verts.push_back(center);
    mesh.topos.push_back(intTopo);

    // Gen new stack vertices
    glm::dvec4 arm = upBase;
    for(int j=0; j<sliceCount; ++j, arm = dSlice * arm)
    {
        double radius = dRadius;
        for(int i=0; i<layerCount; ++i, radius += dRadius)
        {
            glm::dvec3 pos = center + glm::dvec3(arm) * radius;
            mesh.verts.push_back(pos);

            mesh.topos.push_back(
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
            mesh.pris.push_back(
                MeshPri(
                    minK,
                    minK + minJ,
                    minK + maxJ,
                    maxK,
                    maxK + minJ,
                    maxK + maxJ
            ));


            // Create hex layers
            for(int i=1; i<layerCount; ++i)
            {
                int maxI = i;
                int minI = i-1;

                mesh.hexs.push_back(
                    MeshHex(
                        minI + minJ + minK,
                        maxI + minJ + minK,
                        maxI + maxJ + minK,
                        minI + maxJ + minK,
                        minI + minJ + maxK,
                        maxI + minJ + maxK,
                        maxI + maxJ + maxK,
                        minI + maxJ + maxK
                ));
            }
        }
    }
}
