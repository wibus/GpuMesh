#include "MultiPosGradDsntSmoother.h"

using namespace cellar;


// Parameters
const int MPGDSecurityCycleCount = 5;
const double MPGDLocalSizeToNodeShift = 1.0 / 25.0;


// CUDA Drivers
void installCudaMultiPosGradDsntSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift);
void installCudaMultiPosGradDsntSmoother()
{
    installCudaMultiPosGradDsntSmoother(
                MPGDSecurityCycleCount,
                MPGDLocalSizeToNodeShift);
}
void smoothCudaMultiPosGradDsntVertices(
        const NodeGroups::GpuDispatch& dispatch);


MultiPosGradDsntSmoother::MultiPosGradDsntSmoother() :
    GradientDescentSmoother(
        {":/glsl/compute/Smoothing/VertexWise/MultiPosGradDsnt.glsl"},
        installCudaMultiPosGradDsntSmoother,
        smoothCudaMultiPosGradDsntVertices)
{

}

MultiPosGradDsntSmoother::~MultiPosGradDsntSmoother()
{

}

void MultiPosGradDsntSmoother::smoothMeshGlsl(
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
            "Mesh won't be touched.", "MultiPosGradDsntSmoother"));
    }
}

void MultiPosGradDsntSmoother::smoothMeshCuda(
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
            "Mesh won't be touched.", "MultiPosGradDsntSmoother"));
    }
}

bool MultiPosGradDsntSmoother::verifyMeshForGpuLimitations(
            const Mesh& mesh) const
{
//    for(const MeshTopo& topo : mesh.topos)
//    {
//        if(topo.neighborElems.size() > ELEMENT_SLOT_COUNT)
//        {
//            getLog().postMessage(new Message('E', false,
//                "Some nodes have too many neighbor elements. "\
//                "Maximum " + std::to_string(ELEMENT_SLOT_COUNT) +
//                ". A node with " + std::to_string(topo.neighborElems.size()) + " found.",
//                "MultiPosGradDsntSmoother"));
//            return false;
//        }
//    }

    return true;
}

std::string MultiPosGradDsntSmoother::glslLauncher() const
{
    return "";
}

NodeGroups::GpuDispatcher MultiPosGradDsntSmoother::glslDispatcher() const
{
    const uint NODE_THREAD_COUNT = 32;
    const uint POSITION_THREAD_COUNT = 8;

    return [](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(POSITION_THREAD_COUNT, NODE_THREAD_COUNT, 1);
        d.workgroupCount = glm::uvec3(
            glm::ceil(double(d.gpuBufferSize)/NODE_THREAD_COUNT), 1, 1);
    };
}

NodeGroups::GpuDispatcher MultiPosGradDsntSmoother::cudaDispatcher() const
{
    const uint NODE_THREAD_COUNT = 4;
    const uint POSITION_THREAD_COUNT = 8;

    return [](NodeGroups::GpuDispatch& d)
    {
        d.workgroupSize = glm::uvec3(POSITION_THREAD_COUNT, NODE_THREAD_COUNT, 1);
        d.workgroupCount = glm::uvec3(
            glm::ceil(double(d.gpuBufferSize)/NODE_THREAD_COUNT), 1, 1);
    };
}
