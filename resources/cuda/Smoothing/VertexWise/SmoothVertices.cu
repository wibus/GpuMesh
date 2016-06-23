#include "Base.cuh"
#include <DataStructures/NodeGroups.h>
#include <Smoothers/AbstractSmoother.h>


__device__ smoothVertFct smoothVert = nullptr;

__device__  float MoveCoeff = 0.35;


// Smoothing Helper
__device__ uint getInvocationVertexId();
__device__ bool isSmoothableVertex(uint vId);


__global__ void smoothVerticesCudaMain()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        smoothVert(vId);
    }
}


// CUDA Drivers
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch);

void smoothCudaVertices(
        const NodeGroups::GpuDispatch& dispatch,
        size_t workgroupSize,
        float moveCoeff)
{
    setupCudaIndependentDispatch(dispatch);
    cudaMemcpyToSymbol(MoveCoeff, &moveCoeff, sizeof(float));

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothVerticesCudaMain<<<dispatch.workgroupCount, workgroupSize>>>();
    cudaCheckErrors("CUDA error during vertices smoothing");

    cudaDeviceSynchronize();
}
