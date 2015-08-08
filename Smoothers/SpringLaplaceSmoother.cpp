#include "SpringLaplaceSmoother.h"

#include "Evaluators/AbstractEvaluator.h"
#include "OptimizationHelper.h"

using namespace std;


SpringLaplaceSmoother::SpringLaplaceSmoother() :
    AbstractSmoother({":/shaders/compute/Smoothing/SpringLaplace.glsl"})
{

}

SpringLaplaceSmoother::~SpringLaplaceSmoother()
{

}

void SpringLaplaceSmoother::smoothVertices(
        Mesh& mesh,
        AbstractEvaluator& evaluator,
        size_t first,
        size_t last,
        bool synchronize)
{
    std::vector<MeshVert>& verts = mesh.vert;
    std::vector<MeshTet>& tets = mesh.tetra;
    std::vector<MeshPri>& pris = mesh.prism;
    std::vector<MeshHex>& hexs = mesh.hexa;

    for(int v = first; v < last; ++v)
    {
        const MeshTopo& topo = mesh.topo[v];
        if(topo.isFixed)
            continue;

        size_t neigElemCount = topo.neighborElems.size();
        if(neigElemCount == 0)
            continue;

        glm::dvec3 patchCenter =
            OptimizationHelper::findPatchCenter(
                v, topo, verts,
                tets, pris, hexs);

        glm::dvec3& pos = mesh.vert[v].p;
        pos = glm::mix(pos, patchCenter, _moveFactor);

        if(topo.isBoundary)
            pos = (*topo.snapToBoundary)(pos);
    }
}
