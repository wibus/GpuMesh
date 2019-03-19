#include "MultiElemNMSmoother.h"

using namespace cellar;


// Parameters
const double MENMValueConvergence = 0.000100;
const int MENMSecurityCycleCount = 12;
const double MENMLocalSizeToNodeShift = 1.0 / 24.0;
const double MENMAlpha = 1.0;
const double MENMBeta = 0.5;
const double MENMGamma = 2.0;
const double MENMDelta = 0.5;

// CUDA Drivers
void installCudaMultiElemNMSmoother(
        float h_valueConvergence,
        int h_securityCycleCount,
        float h_localSizeToNodeShift,
        float h_alpha,
        float h_beta,
        float h_gamma,
        float h_delta);
void installCudaMultiElemNMSmoother()
{
    installCudaMultiElemNMSmoother(
        MENMValueConvergence,
        MENMSecurityCycleCount,
        MENMLocalSizeToNodeShift,
        MENMAlpha,
        MENMBeta,
        MENMGamma,
        MENMDelta);
}
void smoothCudaMultiElemNMVertices(
        const NodeGroups::GpuDispatch& dispatch);


const int MultiElemNMSmoother::ELEMENT_THREAD_COUNT = 8;

const int MultiElemNMSmoother::ELEMENT_PER_THREAD_COUNT =
        96 / MultiElemNMSmoother::ELEMENT_THREAD_COUNT;


MultiElemNMSmoother::MultiElemNMSmoother() :
    NelderMeadSmoother(
        {":/glsl/compute/Smoothing/VertexWise/MultiElemNM.glsl"},
        installCudaMultiElemNMSmoother,
        smoothCudaMultiElemNMVertices)
{

}

MultiElemNMSmoother::~MultiElemNMSmoother()
{

}

void MultiElemNMSmoother::smoothMeshGlsl(
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
            "Mesh won't be touched.", "MultiElemNMSmoother"));
    }
}

void MultiElemNMSmoother::smoothMeshCuda(
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
            "Mesh won't be touched.", "MultiElemNMSmoother"));
    }
}

bool MultiElemNMSmoother::verifyMeshForGpuLimitations(
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
                "MultiElemNMSmoother"));
            return false;
        }
    }

    return true;
}

std::string MultiElemNMSmoother::glslLauncher() const
{
    return "";
}

NodeGroups::GpuDispatcher MultiElemNMSmoother::glslDispatcher() const
{
    return [](NodeGroups::GpuDispatch& d)
    {
		const uint NODE_THREAD_COUNT = 4;

        d.workgroupSize = glm::uvec3(ELEMENT_THREAD_COUNT, NODE_THREAD_COUNT, 1);
        d.workgroupCount = glm::uvec3(
            glm::ceil(double(d.gpuBufferSize)/NODE_THREAD_COUNT), 1, 1);
    };
}

NodeGroups::GpuDispatcher MultiElemNMSmoother::cudaDispatcher() const
{
    return [](NodeGroups::GpuDispatch& d)
    {
		const uint NODE_THREAD_COUNT = 4;

        d.workgroupSize = glm::uvec3(ELEMENT_THREAD_COUNT, NODE_THREAD_COUNT, 1);
        d.workgroupCount = glm::uvec3(
            glm::ceil(double(d.gpuBufferSize)/NODE_THREAD_COUNT), 1, 1);
    };
}
