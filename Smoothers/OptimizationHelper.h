#ifndef GPUMESH_OPTIMIZATIONHELPER
#define GPUMESH_OPTIMIZATIONHELPER

#include <string>
#include <vector>

#include <GLM/glm.hpp>

class Mesh;
struct MeshTopo;

class AbstractEvaluator;


class OptimizationHelper
{
private:
    OptimizationHelper();

public:
    static std::string shaderName();

    static glm::dvec3 computePatchCenter(
            const Mesh& mesh,
            size_t vertId,
            const MeshTopo& topo);

    static void accumulatePatchQuality(double elemQ, double& patchQ);
    static void finalizePatchQuality(double& patchQ);
    static double computePatchQuality(
            const Mesh& mesh,
            const MeshTopo& topo,
            const AbstractEvaluator& evaluator);
};

#endif // GPUMESH_OPTIMIZATIONHELPER
