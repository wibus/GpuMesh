#ifndef GPUMESH_OPTIMIZATIONHELPER
#define GPUMESH_OPTIMIZATIONHELPER

#include <string>
#include <vector>

#include <GLM/glm.hpp>

class Mesh;
struct MeshTopo;

class AbstractEvaluator;
class AbstractDiscretizer;


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

    static glm::dvec3 computeSpringForce(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& pi,
            const glm::dvec3& pj);
    static glm::dvec3 computePatchCenter(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
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
};

#endif // GPUMESH_OPTIMIZATIONHELPER
