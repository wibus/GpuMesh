#include "MultiElemGradDsntSmoother.h"

using namespace cellar;


// Parameters
const int MEGDSecurityCycleCount = 5;
const double MEGDLocalSizeToNodeShift = 1.0 / 25.0;


// CUDA Drivers
void installCudaMultiElemGradDsntSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift);
void installCudaMultiElemGradDsntSmoother()
{
    installCudaMultiElemGradDsntSmoother(
                MEGDSecurityCycleCount,
                MEGDLocalSizeToNodeShift);
}
void smoothCudaMultiElemGradDsntVertices(
        const NodeGroups::GpuDispatch& dispatch);


const int MultiElemGradDsntSmoother::NODE_THREAD_COUNT = 4;
const int MultiElemGradDsntSmoother::ELEMENT_THREAD_COUNT = 8;
const int MultiElemGradDsntSmoother::POSITION_SLOT_COUNT = 8;

const int MultiElemGradDsntSmoother::ELEMENT_PER_THREAD_COUNT =
        96 / MultiElemGradDsntSmoother::ELEMENT_THREAD_COUNT;

MultiElemGradDsntSmoother::MultiElemGradDsntSmoother() :
    GradientDescentSmoother(
        {":/glsl/compute/Smoothing/VertexWise/MultiElemGradDsnt.glsl"},
        installCudaMultiElemGradDsntSmoother)
{

}

MultiElemGradDsntSmoother::~MultiElemGradDsntSmoother()
{

}

void MultiElemGradDsntSmoother::smoothMeshGlsl(
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
            "Mesh won't be touched.", "MultiElemGradDsntSmoother"));
    }
}

void MultiElemGradDsntSmoother::smoothMeshCuda(
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
            "Mesh won't be touched.", "MultiElemGradDsntSmoother"));
    }
}

bool MultiElemGradDsntSmoother::verifyMeshForGpuLimitations(
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
                "MultiElemGradDsntSmoother"));
            return false;
        }
    }

    return true;
}

void MultiElemGradDsntSmoother::launchCudaKernel(
            const NodeGroups::GpuDispatch& dispatch)
{
    smoothCudaMultiElemGradDsntVertices(dispatch);
}

std::string MultiElemGradDsntSmoother::glslLauncher() const
{
    return "";
}

size_t MultiElemGradDsntSmoother::glslNodesPerBlock() const
{
    return 32;
}

size_t MultiElemGradDsntSmoother::cudaNodesPerBlock() const
{
    return 4;
}
