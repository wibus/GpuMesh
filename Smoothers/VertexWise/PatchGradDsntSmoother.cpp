#include "PatchGradDsntSmoother.h"

using namespace cellar;


// Parameters
const int PGDSecurityCycleCount = 5;
const double PGDLocalSizeToNodeShift = 1.0 / 25.0;


// CUDA Drivers
void installCudaPatchGradDsntSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift);
void installCudaPatchGradDsntSmoother()
{
    installCudaPatchGradDsntSmoother(
                PGDSecurityCycleCount,
                PGDLocalSizeToNodeShift);
}
void smoothCudaPatchGradDsntVertices(
        const NodeGroups::GpuDispatch& dispatch);


const int PatchGradDsntSmoother::POSITION_THREAD_COUNT = 8;
const int PatchGradDsntSmoother::ELEMENT_THREAD_COUNT = 8;

const int PatchGradDsntSmoother::ELEMENT_PER_THREAD_COUNT =
        96 / PatchGradDsntSmoother::ELEMENT_THREAD_COUNT;

PatchGradDsntSmoother::PatchGradDsntSmoother() :
    GradientDescentSmoother(
        {":/glsl/compute/Smoothing/VertexWise/PatchGradDsnt.glsl"},
        installCudaPatchGradDsntSmoother,
        smoothCudaPatchGradDsntVertices)
{

}

PatchGradDsntSmoother::~PatchGradDsntSmoother()
{

}

void PatchGradDsntSmoother::smoothMeshGlsl(
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
            "Mesh won't be touched.", "PatchGradDsntSmoother"));
    }
}

void PatchGradDsntSmoother::smoothMeshCuda(
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
            "Mesh won't be touched.", "PatchGradDsntSmoother"));
    }
}

bool PatchGradDsntSmoother::verifyMeshForGpuLimitations(
            const Mesh& mesh) const
{
    for(const MeshTopo& topo : mesh.topos)
    {
        const uint ELEMENT_SLOT_COUNT =
                ELEMENT_PER_THREAD_COUNT *
                ELEMENT_THREAD_COUNT;

        if(topo.neighborElems.size() > ELEMENT_SLOT_COUNT)
        {
            getLog().postMessage(new Message('E', false,
                "Some nodes have too many neighbor elements. "\
                "Maximum " + std::to_string(ELEMENT_SLOT_COUNT) +
                ". A node with " + std::to_string(topo.neighborElems.size()) + " found.",
                "PatchGradDsntSmoother"));
            return false;
        }
    }

    return true;
}

std::string PatchGradDsntSmoother::glslLauncher() const
{
    return "";
}

NodeGroups::GpuDispatcher PatchGradDsntSmoother::glslDispatcher() const
{
    const uint POSITION_THREAD_COUNT = 8;
    const uint ELEMENT_THREAD_COUNT = 32;

    return [](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(POSITION_THREAD_COUNT, ELEMENT_THREAD_COUNT, 1);
        d.workgroupCount = glm::uvec3(d.gpuBufferSize, 1, 1);
    };
}

NodeGroups::GpuDispatcher PatchGradDsntSmoother::cudaDispatcher() const
{
    const uint POSITION_THREAD_COUNT = 8;
    const uint ELEMENT_THREAD_COUNT = 8;

    return [](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(POSITION_THREAD_COUNT, ELEMENT_THREAD_COUNT, 1);
        d.workgroupCount = glm::uvec3(d.gpuBufferSize, 1, 1);
    };
}
