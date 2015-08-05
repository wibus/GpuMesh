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
struct MeshNeigElem;
struct MeshNeigVert;

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
            const std::vector<MeshNeigElem>& neighborElems,
            const std::vector<MeshTet>& tets,
            const std::vector<MeshPri>& pris,
            const std::vector<MeshHex>& hexs);

    static double findLocalElementSize(size_t v,
            const std::vector<MeshVert>& verts,
            const std::vector<MeshNeigVert>& neigVerts);


    static void integrateQuality(double& total, double shape);

    static void testTetPropositions(
            Mesh& mesh,
            uint vertId,
            const MeshTet& elem,
            AbstractEvaluator& evaluator,
            glm::dvec3 propositions[],
            double propQualities[],
            uint propositionCount);

    static void testPriPropositions(
            Mesh& mesh,
            uint vertId,
            const MeshPri& elem,
            AbstractEvaluator& evaluator,
            glm::dvec3 propositions[],
            double propQualities[],
            uint propositionCount);

    static void testHexPropositions(
            Mesh& mesh,
            uint vertId,
            const MeshHex& elem,
            AbstractEvaluator& evaluator,
            glm::dvec3 propositions[],
            double propQualities[],
            uint propositionCount);

    static void computePropositionPatchQualities(
            Mesh& mesh,
            uint vertId,
            const MeshTopo& topo,
            const std::vector<MeshNeigElem>& neighborElems,
            const std::vector<MeshTet>& tets,
            const std::vector<MeshPri>& pris,
            const std::vector<MeshHex>& hexs,
            AbstractEvaluator& evaluator,
            glm::dvec3 propositions[],
            double patchQualities[],
            uint propositionCount);
};

#endif // GPUMESH_OPTIMIZATIONHELPER
