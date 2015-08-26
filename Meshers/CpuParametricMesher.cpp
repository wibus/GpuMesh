#include "CpuParametricMesher.h"

#include <GLM/gtc/matrix_transform.hpp>

#include <CellarWorkbench/Misc/Log.h>

using namespace std;
using namespace cellar;


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


CpuParametricMesher::CpuParametricMesher() :
    _pipeSurface(new PipeSurfaceBoundary()),
    _pipeExtFace(new PipeExtFaceBoundary()),
    _pipeExtEdge(new PipeExtEdgeBoundary())
{
    using namespace std::placeholders;
    _modelFuncs.setDefault("Pipe");
    _modelFuncs.setContent({
        {string("Pipe"),   ModelFunc(bind(&CpuParametricMesher::genPipe,   this, _1, _2))},
        {string("Bottle"), ModelFunc(bind(&CpuParametricMesher::genBottle, this, _1, _2))},
        {string("Squish"), ModelFunc(bind(&CpuParametricMesher::genSquish, this, _1, _2))},
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
                    PIPE_RADIUS,
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
               PIPE_RADIUS,
               arcPipeStackCount,
               sliceCount,
               layerCount,
               false,
               false);

    insertStraightRingPipe(mesh,
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

    mesh.setmodelBoundariesShaderName(
        ":/shaders/compute/Boundary/ElbowPipe.glsl");
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
                    lastSliceBaseIdx + currRingBaseIdx + currRingSideBaseIdx,
                    currSliceBaseIdx + currRingBaseIdx + currRingSideBaseIdx,
                    lastSliceBaseIdx + currRingBaseIdx + currRingSideNextIdx,
                    currSliceBaseIdx + currRingBaseIdx + currRingSideNextIdx,
                    lastSliceBaseIdx + lastRingBaseIdx + lastRingSideBaseIdx,
                    currSliceBaseIdx + lastRingBaseIdx + lastRingSideBaseIdx));

                size_t currRingSideStepIdx = currRingSideNextIdx;
                size_t lastRingSideStepIdx = lastRingSideBaseIdx;
                for(int v=1; v < r; ++v)
                {
                    currRingSideNextIdx = (currRingSideStepIdx + 1)%currRingVertCount;
                    size_t lastRingSideNextIdx = (lastRingSideStepIdx + 1)%lastRingVertCount;

                    mesh.pris.push_back(MeshPri(
                        lastSliceBaseIdx + lastRingBaseIdx + lastRingSideStepIdx,
                        currSliceBaseIdx + lastRingBaseIdx + lastRingSideStepIdx,
                        lastSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        currSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        lastSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx,
                        currSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx));

                    mesh.pris.push_back(MeshPri(
                        lastSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx,
                        currSliceBaseIdx + lastRingBaseIdx + lastRingSideNextIdx,
                        lastSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        currSliceBaseIdx + currRingBaseIdx + currRingSideStepIdx,
                        lastSliceBaseIdx + currRingBaseIdx + currRingSideNextIdx,
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
                    currSliceBaseIdx + nextD,
                    lastSliceBaseIdx + nextD,
                    currStackBaseIdx + d,
                    lastStackBaseIdx + d,
                    currStackBaseIdx + nextD,
                    lastStackBaseIdx + nextD));
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
                    currStackBaseIdx + lastRingIdx + nextD,
                    lastStackBaseIdx + lastRingIdx + nextD,
                    currStackBaseIdx + currRingIdx + d,
                    lastStackBaseIdx + currRingIdx + d,
                    currStackBaseIdx + currRingIdx + nextD,
                    lastStackBaseIdx + currRingIdx + nextD));
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
                    currStackBaseIdx + ultimateRingIdx + nextD,
                    lastStackBaseIdx + ultimateRingIdx + nextD,
                    currRingIdx + d,
                    lastRingIdx + d,
                    currRingIdx + nextD,
                    lastRingIdx + nextD));
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
                    currStackBaseIdx + lastRingIdx + nextD,
                    lastStackBaseIdx + lastRingIdx + nextD,
                    currStackBaseIdx + currRingIdx + d,
                    lastStackBaseIdx + currRingIdx + d,
                    currStackBaseIdx + currRingIdx + nextD,
                    lastStackBaseIdx + currRingIdx + nextD));
            }
        }

        lastStackBaseIdx = currStackBaseIdx;
    }
}

void CpuParametricMesher::genSquish(Mesh& mesh, size_t vertexCount)
{
    double squishRadius = 0.3;
    double squishHeight = 0.6;

    const int pow1_3 =  glm::pow((double)vertexCount, 1.0/3.0);
    const int pow1_3_pair = ((pow1_3 + 1) / 2) * 2;
    const int X_COUNT = pow1_3_pair;
    const int Y_COUNT = pow1_3_pair;
    const int Z_COUNT = pow1_3_pair;
    for(int z=-Z_COUNT/2; z <= Z_COUNT/2; ++z)
    {
        for(int y=-Y_COUNT/2; y <= Y_COUNT/2; ++y)
        {
            for(int x=-X_COUNT/2; x <= X_COUNT/2; ++x)
            {
                glm::dvec2 arm;
                double radius = glm::max(glm::abs(x), glm::abs(y)) * (2.0 * squishRadius / Y_COUNT);
                if(radius != 0.0) arm = glm::normalize(glm::dvec2(x, y)) * radius;

                mesh.verts.push_back(glm::dvec3(
                    arm, z * squishHeight / Z_COUNT));
            }
        }
    }

    const int X_WIDTH = 1;
    const int Y_WIDTH = X_COUNT + 1;
    const int Z_WIDTH = (Y_COUNT+1) * Y_WIDTH;
    for(int z=0; z < Z_COUNT; ++z)
    {
        for(int y=0; y< Y_COUNT; ++y)
        {
            for(int x=0; x< X_COUNT; ++x)
            {
                MeshHex hex(
                    (x+0) * X_WIDTH + (y+0) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+0) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+0) * X_WIDTH + (y+1) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+1) * Y_WIDTH + (z+0) * Z_WIDTH,
                    (x+0) * X_WIDTH + (y+0) * Y_WIDTH + (z+1) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+0) * Y_WIDTH + (z+1) * Z_WIDTH,
                    (x+0) * X_WIDTH + (y+1) * Y_WIDTH + (z+1) * Z_WIDTH,
                    (x+1) * X_WIDTH + (y+1) * Y_WIDTH + (z+1) * Z_WIDTH);
                mesh.hexs.push_back(hex);
            }
        }
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
    MeshTopo extTopo(isBoundary ? _pipeExtEdge.get() : _pipeSurface.get());
    MeshTopo intTopo(isBoundary ? _pipeExtFace.get() : &MeshTopo::NO_BOUNDARY);

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

                mesh.hexs.push_back(
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
