#include "SpawnSearchSmoother.h"

#include <CellarWorkbench/Misc/Distribution.h>

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;

const double SSMoveCoeff = 0.10;


// CUDA Drivers
void installCudaSpawnSearchSmoother(float moveCoeff);
void installCudaSpawnSearchSmoother()
{
    installCudaSpawnSearchSmoother(SSMoveCoeff);
}


const int SpawnSearchSmoother::PROPOSITION_COUNT = 64;


SpawnSearchSmoother::SpawnSearchSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/SpawnSearch.glsl"},
        installCudaSpawnSearchSmoother)
{
    _offsets.clear();

    for(int k=0; k<=4; ++k)
        for(int j=0; j<=4; ++j)
            for(int i=0; i<=4; ++i)
                _offsets.push_back(glm::dvec3(i, j, k) - glm::dvec3(1.5));

    _offsets[0] = glm::dvec3(0, 0, 0);
}

SpawnSearchSmoother::~SpawnSearchSmoother()
{

}

void SpawnSearchSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
}

void SpawnSearchSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plot);
    plot.addSmoothingProperty("Method Name", "Spread Search");
}

void SpawnSearchSmoother::smoothVertices(
        Mesh& mesh,
        const MeshCrew& crew,
        const std::vector<uint>& vIds)
{
    std::vector<MeshVert>& verts = mesh.verts;
    const vector<MeshTopo>& topos = mesh.topos;

    size_t vIdCount = vIds.size();
    for(int v = 0; v < vIdCount; ++v)
    {
        uint vId = vIds[v];

        glm::dvec3& pos = verts[vId].p;

        // Compute local element size
        double localSize = SSMoveCoeff *
            crew.measurer().computeLocalElementSize(mesh, vId);


        // Define propositions for new vertex's position
        glm::dvec3 propositions[PROPOSITION_COUNT];
        for(int p=0; p < PROPOSITION_COUNT; ++p)
        {
            propositions[p] = pos + glm::dvec3(_offsets[p]) * localSize;
        }

        const MeshTopo& topo = topos[vId];
        if(topo.snapToBoundary->isConstrained())
        {
            for(uint p=0; p < PROPOSITION_COUNT; ++p)
                propositions[p] = (*topo.snapToBoundary)(propositions[p]);
        }


        // Choose best position
        uint bestProposition = 0;
        double bestQualityMean = -numeric_limits<double>::infinity();
        for(uint p=0; p < PROPOSITION_COUNT; ++p)
        {
            // Since 'pos' is a reference on vertex's position
            // modifing its value here should be seen by the evaluator
            pos = propositions[p];

            // Compute patch quality
            double patchQuality =
                crew.evaluator().patchQuality(
                    mesh, crew.sampler(), crew.measurer(), vId);

            if(patchQuality > bestQualityMean)
            {
                bestQualityMean = patchQuality;
                bestProposition = p;
            }
        }


        // Update vertex's position
        pos = propositions[bestProposition];
    }
}
