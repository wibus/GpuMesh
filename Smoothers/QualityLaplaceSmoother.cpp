#include "QualityLaplaceSmoother.h"

#include <iostream>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;


QualityLaplaceSmoother::QualityLaplaceSmoother() :
    AbstractSmoother(":/shaders/compute/Smoothing/QualityLaplace.glsl")
{

}

QualityLaplaceSmoother::~QualityLaplaceSmoother()
{

}

const uint PROPOSITION_COUNT = 4;

inline void integrateQuality(double& total, double shape)
{
    total *= shape;
}

void testTetPropositions(
        uint vertId,
        Mesh& mesh,
        MeshTet& elem,
        AbstractEvaluator& evaluator,
        glm::dvec3 propositions[],
        double propQualities[])
{
    // Extract element's vertices
    glm::dvec3 vp[MeshTet::VERTEX_COUNT] = {
        mesh.vert[elem[0]],
        mesh.vert[elem[1]],
        mesh.vert[elem[2]],
        mesh.vert[elem[3]],
    };

    // Find Vertex position in element
    uint elemVertId = 0;
    if(vertId == elem[1])
        elemVertId = 1;
    else if(vertId == elem[2])
        elemVertId = 2;
    else if(vertId == elem[3])
        elemVertId = 3;

    for(uint p=0; p < PROPOSITION_COUNT; ++p)
    {
        if(propQualities[p] > 0.0)
        {
            vp[elemVertId] = propositions[p];

            integrateQuality(
                propQualities[p],
                evaluator.tetQuality(vp));
        }
    }
}

void testPriPropositions(
        uint vertId,
        Mesh& mesh,
        MeshPri& elem,
        AbstractEvaluator& evaluator,
        glm::dvec3 propositions[],
        double propQualities[])
{
    // Extract element's vertices
    glm::dvec3 vp[MeshPri::VERTEX_COUNT] = {
        mesh.vert[elem[0]],
        mesh.vert[elem[1]],
        mesh.vert[elem[2]],
        mesh.vert[elem[3]],
        mesh.vert[elem[4]],
        mesh.vert[elem[5]],
    };

    // Find Vertex position in element
    uint elemVertId = 0;
    if(vertId == elem[1])
        elemVertId = 1;
    else if(vertId == elem[2])
        elemVertId = 2;
    else if(vertId == elem[3])
        elemVertId = 3;
    else if(vertId == elem[4])
        elemVertId = 4;
    else if(vertId == elem[5])
        elemVertId = 5;

    for(uint p=0; p < PROPOSITION_COUNT; ++p)
    {
        if(propQualities[p] > 0.0)
        {
            vp[elemVertId] = propositions[p];

            integrateQuality(
                propQualities[p],
                evaluator.priQuality(vp));
        }
    }
}

void testHexPropositions(
        uint vertId,
        Mesh& mesh,
        MeshHex& elem,
        AbstractEvaluator& evaluator,
        glm::dvec3 propositions[],
        double propQualities[])
{
    // Extract element's vertices
    glm::dvec3 vp[MeshHex::VERTEX_COUNT] = {
        mesh.vert[elem[0]],
        mesh.vert[elem[1]],
        mesh.vert[elem[2]],
        mesh.vert[elem[3]],
        mesh.vert[elem[4]],
        mesh.vert[elem[5]],
        mesh.vert[elem[6]],
        mesh.vert[elem[7]],
    };

    // Find vertex position in element
    uint elemVertId = 0;
    if(vertId == elem[1])
        elemVertId = 1;
    else if(vertId == elem[2])
        elemVertId = 2;
    else if(vertId == elem[3])
        elemVertId = 3;
    else if(vertId == elem[4])
        elemVertId = 4;
    else if(vertId == elem[5])
        elemVertId = 5;
    else if(vertId == elem[6])
        elemVertId = 6;
    else if(vertId == elem[7])
        elemVertId = 7;

    for(uint p=0; p < PROPOSITION_COUNT; ++p)
    {
        if(propQualities[p] > 0.0)
        {
            vp[elemVertId] = propositions[p];

            integrateQuality(
                propQualities[p],
                evaluator.hexQuality(vp));
        }
    }
}

void QualityLaplaceSmoother::smoothMeshCpp(
        Mesh& mesh,
        AbstractEvaluator& evaluator)
{
    std::vector<MeshVert>& verts = mesh.vert;
    std::vector<MeshTet>& tets = mesh.tetra;
    std::vector<MeshPri>& pris = mesh.prism;
    std::vector<MeshHex>& hexs = mesh.hexa;


    _smoothPassId = 0;
    while(evaluateCpuMeshQuality(mesh, evaluator))
    {
        uint vertCount = verts.size();
        for(uint v = 0; v < vertCount; ++v)
        {
            const MeshTopo& topo = mesh.topo[v];
            if(topo.isFixed)
                continue;

            const vector<MeshNeigElem>& neighborElems = topo.neighborElems;
            uint neigElemCount = neighborElems.size();
            if(neigElemCount == 0)
                continue;

            // Compute patch center
            uint totalVertCount = 0;
            glm::dvec3 patchCenter(0.0);
            for(uint n=0; n < neigElemCount; ++n)
            {
                const MeshNeigElem& neigElem = neighborElems[n];
                switch(neigElem.type)
                {
                case MeshTet::ELEMENT_TYPE:
                    totalVertCount += MeshTet::VERTEX_COUNT - 1;
                    for(uint i=0; i < MeshTet::VERTEX_COUNT; ++i)
                        patchCenter += verts[tets[neigElem.id].v[i]].p;
                    break;

                case MeshPri::ELEMENT_TYPE:
                    totalVertCount += MeshPri::VERTEX_COUNT - 1;
                    for(uint i=0; i < MeshPri::VERTEX_COUNT; ++i)
                        patchCenter += verts[pris[neigElem.id].v[i]].p;
                    break;

                case MeshHex::ELEMENT_TYPE:
                    totalVertCount += MeshHex::VERTEX_COUNT - 1;
                    for(uint i=0; i < MeshHex::VERTEX_COUNT; ++i)
                        patchCenter += verts[hexs[neigElem.id].v[i]].p;
                    break;
                }
            }

            glm::dvec3& pos = verts[v].p;
            patchCenter = (patchCenter - pos * double(neigElemCount))
                            / double(totalVertCount);
            glm::dvec3 centerDist = patchCenter - pos;


            // Define propositions for new vertex's position
            glm::dvec3 propositions[PROPOSITION_COUNT] = {
                pos,
                patchCenter - centerDist * _moveFactor,
                patchCenter,
                patchCenter + centerDist * _moveFactor,
            };

            if(topo.isBoundary)
                for(uint p=1; p < PROPOSITION_COUNT; ++p)
                    propositions[p] = topo.snapToBoundary(propositions[p]);

            double patchQualities[PROPOSITION_COUNT] = {1.0, 1.0, 1.0, 1.0};


            // Compute proposition's patch quality
            for(uint n=0; n < neigElemCount; ++n)
            {
                const MeshNeigElem& neigElem = topo.neighborElems[n];
                switch(neigElem.type)
                {
                case MeshTet::ELEMENT_TYPE:
                    testTetPropositions(
                        v,
                        mesh,
                        tets[neigElem.id],
                        evaluator,
                        propositions,
                        patchQualities);
                    break;

                case MeshPri::ELEMENT_TYPE:
                    testPriPropositions(
                        v,
                        mesh,
                        pris[neigElem.id],
                        evaluator,
                        propositions,
                        patchQualities);
                    break;

                case MeshHex::ELEMENT_TYPE:
                    testHexPropositions(
                        v,
                        mesh,
                        hexs[neigElem.id],
                        evaluator,
                        propositions,
                        patchQualities);
                    break;
                }
            }

            // Find best proposition based on patch quality
            uint bestProposition = 0;
            double bestQualityResult = patchQualities[0];
            for(uint p=1; p < PROPOSITION_COUNT; ++p)
            {
                if(bestQualityResult < patchQualities[p])
                {
                    bestProposition = p;
                    bestQualityResult = patchQualities[p];
                }
            }

            // Update vertex's position
            pos = propositions[bestProposition];
        }
    }


    mesh.updateGpuVertices();
}
