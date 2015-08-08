#include "OptimizationHelper.h"

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


std::string OptimizationHelper::shaderName()
{
    return ":/shaders/compute/Smoothing/OptimizationHelper.glsl";
}

double OptimizationHelper::computeLocalElementSize(
        const Mesh& mesh,
        size_t vId)
{
    const std::vector<MeshVert>& verts = mesh.vert;

    const glm::dvec3& pos = verts[vId].p;
    const MeshTopo& topo = mesh.topo[vId];
    const vector<MeshNeigVert>& neigVerts = topo.neighborVerts;

    double totalSize = 0.0;
    size_t neigVertCount = neigVerts.size();
    for(size_t n=0; n < neigVertCount; ++n)
    {
        totalSize += glm::length(pos - verts[neigVerts[n].v].p);
    }

    return totalSize / neigVertCount;
}

glm::dvec3 OptimizationHelper::computePatchCenter(
        const Mesh& mesh,
        size_t vId)
{
    const std::vector<MeshVert>& verts = mesh.vert;
    const std::vector<MeshTet>& tets = mesh.tetra;
    const std::vector<MeshPri>& pris = mesh.prism;
    const std::vector<MeshHex>& hexs = mesh.hexa;

    const MeshTopo& topo = mesh.topo[vId];

    uint totalVertCount = 0;
    glm::dvec3 patchCenter(0.0);
    uint neigElemCount = topo.neighborElems.size();
    for(uint n=0; n < neigElemCount; ++n)
    {
        const MeshNeigElem& neigElem = topo.neighborElems[n];
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

    const glm::dvec3& pos = verts[vId].p;
    patchCenter = (patchCenter - pos * double(neigElemCount))
                    / double(totalVertCount);

    return patchCenter;
}

inline void OptimizationHelper::accumulatePatchQuality(
        double elemQ,
        double& patchQ)
{
    patchQ *= elemQ;
}

inline void OptimizationHelper::finalizePatchQuality(
        double& patchQ)
{
    // no-op
}

double OptimizationHelper::computePatchQuality(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            size_t vId)
{
    const std::vector<MeshTet>& tets = mesh.tetra;
    const std::vector<MeshPri>& pris = mesh.prism;
    const std::vector<MeshHex>& hexs = mesh.hexa;

    const MeshTopo& topo = mesh.topo[vId];

    size_t neigElemCount = topo.neighborElems.size();

    double patchQuality = 1.0;
    for(size_t n=0; n < neigElemCount; ++n)
    {
        const MeshNeigElem& neigElem = topo.neighborElems[n];

        switch(neigElem.type)
        {
        case MeshTet::ELEMENT_TYPE:
            accumulatePatchQuality(
                evaluator.tetQuality(mesh, tets[neigElem.id]),
                patchQuality);
            break;

        case MeshPri::ELEMENT_TYPE:
            accumulatePatchQuality(
                evaluator.priQuality(mesh, pris[neigElem.id]),
                patchQuality);
            break;

        case MeshHex::ELEMENT_TYPE:
            accumulatePatchQuality(
                evaluator.hexQuality(mesh, hexs[neigElem.id]),
                patchQuality);
            break;
        }

        if(patchQuality <= 0.0)
        {
            break;
        }
    }

    finalizePatchQuality(patchQuality);

    return patchQuality;
}
