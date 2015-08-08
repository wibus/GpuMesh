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
        const MeshTopo& topo,
        const vector<MeshVert>& verts,
        const vector<MeshTet>& tets,
        const vector<MeshPri>& pris,
        const vector<MeshHex>& hexs)
{
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

    const glm::dvec3& pos = verts[v].p;
    patchCenter = (patchCenter - pos * double(neigElemCount))
                    / double(totalVertCount);

    return patchCenter;
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
