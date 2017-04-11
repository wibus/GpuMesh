#include "SpawnSearchSmoother.h"

#include <CellarWorkbench/Misc/Log.h>
#include <CellarWorkbench/Misc/Distribution.h>

#include "Boundaries/Constraints/AbstractConstraint.h"
#include "DataStructures/MeshCrew.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Measurers/AbstractMeasurer.h"

using namespace std;
using namespace cellar;

const double SSMoveCoeff = 1 / 20.0;
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


const int SpawnSearchSmoother::SPAWN_COUNT = 64;


SpawnSearchSmoother::SpawnSearchSmoother() :
    AbstractVertexWiseSmoother(
        {":/glsl/compute/Smoothing/VertexWise/SpawnSearch.glsl"},
        installCudaSpawnSearchSmoother,
        smoothCudaSpawnVertices),
    _offsetsSsbo(0)
{
    g_offsets.clear();

    for(int k=0; k<4; ++k)
    {
        for(int j=0; j<4; ++j)
        {
            for(int i=0; i<4; ++i)
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

void SpawnSearchSmoother::smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew)
{
    if(verifyMeshForGpuLimitations(mesh))
    {
        AbstractVertexWiseSmoother::smoothMeshGlsl(mesh, crew);
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Mesh won't be touched.", "SpawnSearchSmoother"));
    }
}

void SpawnSearchSmoother::smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew)
{
    if(verifyMeshForGpuLimitations(mesh))
    {
        AbstractVertexWiseSmoother::smoothMeshCuda(mesh, crew);
    }
    else
    {
        getLog().postMessage(new Message('E', false,
            "Mesh won't be touched.", "SpawnSearchSmoother"));
    }
}

bool SpawnSearchSmoother::verifyMeshForGpuLimitations(
            const Mesh& mesh) const
{
//    if(!mesh.pris.empty())
//    {
//        getLog().postMessage(new Message('E', false,
//            "This mesh contains prismatic elements. "\
//            "This smoother does not support this type of element yet.",
//            "SpawnSearchSmoother"));
//        return false;
//    }

//    if(!mesh.hexs.empty())
//    {
//        getLog().postMessage(new Message('E', false,
//            "This mesh contains hexahedral elements. "\
//            "This smoother does not support this type of element yet.",
//            "SpawnSearchSmoother"));
//        return false;
//    }

//    for(const MeshTopo& topo : mesh.topos)
//    {
//        if(topo.neighborElems.size() > SPAWN_COUNT)
//        {
//            getLog().postMessage(new Message('E', false,
//                "Some nodes have too many neighbor elements. "\
//                "Maximum " + std::to_string(SPAWN_COUNT) +
//                ". A node with " + std::to_string(topo.neighborElems.size()) + " found.",
//                "SpawnSearchSmoother"));
//            return false;
//        }
//    }

    return true;
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
        OptimizationImpl& plotImpl) const
{
    AbstractVertexWiseSmoother::printOptimisationParameters(mesh, plotImpl);
    plotImpl.addSmoothingProperty("Method Name", "Spawn Search");
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


        for(int iter=0; iter < 2; ++iter)
        {
            // Define propositions for new vertex's position
            glm::dvec3 propositions[SPAWN_COUNT];
            for(int p=0; p < SPAWN_COUNT; ++p)
            {
                propositions[p] = pos + glm::dvec3(g_offsets[p]) * scale;
            }

            const MeshTopo& topo = topos[vId];
            if(topo.snapToBoundary->isConstrained())
            {
                for(uint p=0; p < SPAWN_COUNT; ++p)
                    propositions[p] = (*topo.snapToBoundary)(propositions[p]);
            }


            // Choose best position
            uint bestProposition = 0;
            double bestQualityMean = -numeric_limits<double>::infinity();
            for(uint p=0; p < SPAWN_COUNT; ++p)
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

            scale /= 3.0;
        }
    }
}

std::string SpawnSearchSmoother::glslLauncher() const
{
    return "";
}

NodeGroups::GpuDispatcher SpawnSearchSmoother::glslDispatcher() const
{
    return [](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(SPAWN_COUNT, 1, 1);
        d.workgroupCount = glm::uvec3(d.gpuBufferSize, 1, 1);
    };
}

NodeGroups::GpuDispatcher SpawnSearchSmoother::cudaDispatcher() const
{
    return [](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(SPAWN_COUNT, 1, 1);
        d.workgroupCount = glm::uvec3(d.gpuBufferSize, 1, 1);
    };
}
