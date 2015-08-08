#ifndef GPUMESH_OPTIMIZATIONHELPER
#define GPUMESH_OPTIMIZATIONHELPER

#include <string>
#include <vector>

#include <GLM/glm.hpp>

class Mesh;
struct MeshVert;
struct MeshTet;
struct MeshPri;
struct MeshHex;
struct MeshTopo;

class AbstractEvaluator;


class OptimizationHelper
{
private:
    OptimizationHelper();

public:
    static std::string shaderName();

    static glm::dvec3 findPatchCenter(
            size_t v,
            const MeshTopo& topo,
            const std::vector<MeshVert>& verts,
            const std::vector<MeshTet>& tets,
            const std::vector<MeshPri>& pris,
            const std::vector<MeshHex>& hexs);


    static void accumulatePatchQuality(double elemQ, double& patchQ);
    static void finalizePatchQuality(double& patchQ);
};

#endif // GPUMESH_OPTIMIZATIONHELPER
