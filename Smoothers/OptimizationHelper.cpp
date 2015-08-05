#include "OptimizationHelper.h"

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


std::string OptimizationHelper::shaderName()
{
    return ":/shaders/compute/Smoothing/OptimizationHelper.glsl";
}

glm::dvec3 OptimizationHelper::findPatchCenter(
        size_t v,
        const vector<MeshVert>& verts,
        const vector<MeshTet>& tets,
        const vector<MeshPri>& pris,
        const vector<MeshHex>& hexs,
        const vector<MeshNeigElem>& neighborElems)
{
    uint totalVertCount = 0;
    glm::dvec3 patchCenter(0.0);
    uint neigElemCount = neighborElems.size();
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

    const glm::dvec3& pos = verts[v].p;
    patchCenter = (patchCenter - pos * double(neigElemCount))
                    / double(totalVertCount);

    return patchCenter;
}


inline void OptimizationHelper::integrateQuality(
        double& total,
        double shape)
{
    total *= shape;
}

void OptimizationHelper::testTetPropositions(
        uint vertId,
        Mesh& mesh,
        MeshTet& elem,
        AbstractEvaluator& evaluator,
        glm::dvec3 propositions[],
        double propQualities[],
        uint propositionCount)
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

    for(uint p=0; p < propositionCount; ++p)
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

void OptimizationHelper::testPriPropositions(
        uint vertId,
        Mesh& mesh,
        MeshPri& elem,
        AbstractEvaluator& evaluator,
        glm::dvec3 propositions[],
        double propQualities[],
        uint propositionCount)
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

    for(uint p=0; p < propositionCount; ++p)
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

void OptimizationHelper::testHexPropositions(
        uint vertId,
        Mesh& mesh,
        MeshHex& elem,
        AbstractEvaluator& evaluator,
        glm::dvec3 propositions[],
        double propQualities[],
        uint propositionCount)
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

    for(uint p=0; p < propositionCount; ++p)
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
