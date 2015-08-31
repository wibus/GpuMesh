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

    static void accumulatePatchQuality(
            double& patchQuality,
            double& patchWeight,
            double elemQuality);
    static double finalizePatchQuality(
            double patchQuality,
            double patchWeight);
    static double computePatchQuality(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            size_t vId);


    static const int DISPATCH_MODE_CLUSTER;
    static const int DISPATCH_MODE_SCATTER;
};

#endif // GPUMESH_OPTIMIZATIONHELPER
