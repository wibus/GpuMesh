#ifndef GPUMESH_OPTIMIZATIONHELPER
#define GPUMESH_OPTIMIZATIONHELPER

#include <string>
#include <vector>

#include <GLM/glm.hpp>

class Mesh;
struct MeshTopo;

class AbstractEvaluator;


class SmoothingHelper
{
private:
    SmoothingHelper();

public:
    static std::string shaderName();

    static bool isSmoothable(
            const Mesh& mesh,
            size_t vId);

    static double computeLocalElementSize(
            const Mesh& mesh,
            size_t vId);

    static glm::dvec3 computePatchCenter(
            const Mesh& mesh,
            size_t vId);

    static void accumulatePatchQuality(double elemQ, double& patchQ);
    static void finalizePatchQuality(double& patchQ);
    static double computePatchQuality(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            size_t vId);
};

#endif // GPUMESH_OPTIMIZATIONHELPER
