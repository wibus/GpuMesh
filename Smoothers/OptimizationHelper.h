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
struct MeshNeigElem;

class AbstractEvaluator;


class OptimizationHelper
{
private:
    OptimizationHelper();

public:
    static std::string shaderName();

    static glm::dvec3 findPatchCenter(
            size_t v,
            const std::vector<MeshVert>& verts,
            const std::vector<MeshTet>& tets,
            const std::vector<MeshPri>& pris,
            const std::vector<MeshHex>& hexs,
            const std::vector<MeshNeigElem>& neighborElems);


    static void integrateQuality(double& total, double shape);

    static void testTetPropositions(
            uint vertId,
            Mesh& mesh,
            MeshTet& elem,
            AbstractEvaluator& evaluator,
            glm::dvec3 propositions[],
            double propQualities[],
            uint propositionCount);

    static void testPriPropositions(
            uint vertId,
            Mesh& mesh,
            MeshPri& elem,
            AbstractEvaluator& evaluator,
            glm::dvec3 propositions[],
            double propQualities[],
            uint propositionCount);

    static void testHexPropositions(
            uint vertId,
            Mesh& mesh,
            MeshHex& elem,
            AbstractEvaluator& evaluator,
            glm::dvec3 propositions[],
            double propQualities[],
            uint propositionCount);
};

#endif // GPUMESH_OPTIMIZATIONHELPER
