#include "SpawnSearchSmoother.h"

#include <CellarWorkbench/Misc/Distribution.h>

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;

const double SSMoveCoeff = 0.10;
std::vector<glm::dvec4> g_offsets;


// CUDA Drivers
void installCudaSpawnSearchSmoother(float moveCoeff,
        const std::vector<glm::vec4> offsetsBuff);
void installCudaSpawnSearchSmoother()
{
    vector<glm::vec4> gpuOffsets(g_offsets.begin(), g_offsets.end());
    installCudaSpawnSearchSmoother(SSMoveCoeff, gpuOffsets);
}
void smoothCudaSpawnVertices(
        const NodeGroups::GpuDispatch& dispatch);


const int SpawnSearchSmoother::PROPOSITION_COUNT = 64;


SpawnSearchSmoother::SpawnSearchSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/SpawnSearch.glsl"},
        installCudaSpawnSearchSmoother),
    _offsetsSsbo(0)
{
    g_offsets.clear();

    for(int k=0; k<=4; ++k)
    {
        for(int j=0; j<=4; ++j)
        {
            for(int i=0; i<=4; ++i)
            {
                glm::dvec3 offset = glm::dvec3(i, j, k) - glm::dvec3(1.5);
                g_offsets.push_back(glm::dvec4(offset, glm::length(offset)));
            }
        }
    }

    g_offsets[0] = glm::dvec4(0, 0, 0, 0);
}

SpawnSearchSmoother::~SpawnSearchSmoother()
{

}

void SpawnSearchSmoother::lauchCudaKernel(
            const NodeGroups::GpuDispatch& dispatch)
{
    smoothCudaSpawnVertices(dispatch);
}

void SpawnSearchSmoother::setVertexProgramUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program)
{
    AbstractVertexWiseSmoother::setVertexProgramUniforms(mesh, program);
    program.setFloat("MoveCoeff", SSMoveCoeff);


    if(_offsetsSsbo == 0)
    {
        glGenBuffers(1, &_offsetsSsbo);
    }

    GLuint offsetsBase = mesh.glBufferBinding(EBufferBinding::SPAWN_OFFSETS_BUFFER_BINDING);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, offsetsBase,  _offsetsSsbo);

    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _offsetsSsbo);
        vector<glm::vec4> gpuOffsets(g_offsets.begin(), g_offsets.end());
        size_t offsetsSize = sizeof(decltype(gpuOffsets.front())) * gpuOffsets.size();
        glBufferData(GL_SHADER_STORAGE_BUFFER, offsetsSize, gpuOffsets.data(), GL_STREAM_COPY);
    }
}

void SpawnSearchSmoother::printOptimisationParameters(
        const Mesh& mesh,
        OptimizationPlot& plot) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plot);
    plot.addSmoothingProperty("Method Name", "Spawn Search");
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
        double localSize = crew.measurer().computeLocalElementSize(mesh, vId);
        double scale = SSMoveCoeff * localSize;


        // Define propositions for new vertex's position
        glm::dvec3 propositions[PROPOSITION_COUNT];
        for(int p=0; p < PROPOSITION_COUNT; ++p)
        {
            propositions[p] = pos + glm::dvec3(g_offsets[p]) * scale;
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

std::string SpawnSearchSmoother::glslLauncher() const
{
    return "";
}

glm::ivec3 SpawnSearchSmoother::layoutWorkgroups(
        const NodeGroups::GpuDispatch& dispatch) const
{
    return glm::ivec3(dispatch.gpuBufferSize, 1, 1);
}
