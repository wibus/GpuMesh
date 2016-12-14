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


const int MultiPosGradDsntSmoother::POSITION_THREAD_COUNT = 8;
const int MultiPosGradDsntSmoother::ELEMENT_SLOT_COUNT = 96;

MultiPosGradDsntSmoother::MultiPosGradDsntSmoother() :
    GradientDescentSmoother(
        {":/glsl/compute/Smoothing/VertexWise/MultiPosGradDsnt.glsl"},
        installCudaMultiPosGradDsntSmoother)
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
    for(const MeshTopo& topo : mesh.topos)
    {
        if(topo.neighborElems.size() > ELEMENT_SLOT_COUNT)
        {
            getLog().postMessage(new Message('E', false,
                "Some nodes have too many neighbor elements. "\
                "Maximum " + std::to_string(ELEMENT_SLOT_COUNT) +
                ". A node with " + std::to_string(topo.neighborElems.size()) + " found.",
                "MultiPosGradDsntSmoother"));
            return false;
        }
    }

    return true;
}

void MultiPosGradDsntSmoother::launchCudaKernel(
            const NodeGroups::GpuDispatch& dispatch)
{
    smoothCudaMultiPosGradDsntVertices(dispatch);
}

std::string MultiPosGradDsntSmoother::glslLauncher() const
{
    return "";
}

size_t MultiPosGradDsntSmoother::nodesPerBlock() const
{
    return 1;
}
