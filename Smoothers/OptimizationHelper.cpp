#include "OptimizationHelper.h"

#include "DataStructures/Mesh.h"
#include "Evaluators/AbstractEvaluator.h"

using namespace std;


std::string OptimizationHelper::shaderName()
{
    return ":/shaders/compute/Smoothing/OptimizationHelper.glsl";
}

glm::dvec3 OptimizationHelper::computePatchCenter(const Mesh& mesh,
        size_t vertId,
        const MeshTopo& topo)
{
    const std::vector<MeshVert>& verts = mesh.vert;
    const std::vector<MeshTet>& tets = mesh.tetra;
    const std::vector<MeshPri>& pris = mesh.prism;
    const std::vector<MeshHex>& hexs = mesh.hexa;

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

    const glm::dvec3& pos = verts[vertId].p;
    patchCenter = (patchCenter - pos * double(neigElemCount))
                    / double(totalVertCount);

    return patchCenter;
}

double OptimizationHelper::computePatchQuality(
            const Mesh& mesh,
            const MeshTopo& topo,
            const AbstractEvaluator& evaluator)
{
    const std::vector<MeshTet>& tets = mesh.tetra;
    const std::vector<MeshPri>& pris = mesh.prism;
    const std::vector<MeshHex>& hexs = mesh.hexa;

    size_t neigElemCount = topo.neighborElems.size();

    double patchQuality = 1.0;
    for(size_t n=0; n < neigElemCount; ++n)
    {
        const MeshNeigElem& neigElem = topo.neighborElems[n];

        switch(neigElem.type)
        {
        case MeshTet::ELEMENT_TYPE:
            OptimizationHelper::accumulatePatchQuality(
                evaluator.tetQuality(mesh, tets[neigElem.id]),
                patchQuality);
            break;

        case MeshPri::ELEMENT_TYPE:
            OptimizationHelper::accumulatePatchQuality(
                evaluator.priQuality(mesh, pris[neigElem.id]),
                patchQuality);
            break;

        case MeshHex::ELEMENT_TYPE:
            OptimizationHelper::accumulatePatchQuality(
                evaluator.hexQuality(mesh, hexs[neigElem.id]),
                patchQuality);
            break;
        }

        if(patchQuality <= 0.0)
        {
            break;
        }
    }

    OptimizationHelper::finalizePatchQuality(
            patchQuality);

    return patchQuality;
}

void OptimizationHelper::accumulatePatchQuality(
        double elemQ,
        double& patchQ)
{
    patchQ *= elemQ;
}

void OptimizationHelper::finalizePatchQuality(
        double& patchQ)
{

}
