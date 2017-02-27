#include "Base.cuh"
#include <DataStructures/NodeGroups.h>
#include <Smoothers/AbstractSmoother.h>


__device__ smoothVertFct smoothVert = nullptr;


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
        const NodeGroups::GpuDispatch& dispatch)
{
    setupCudaIndependentDispatch(dispatch);

    cudaFuncSetCacheConfig(smoothVerticesCudaMain, cudaFuncCachePreferL1);

    dim3 blockDim(dispatch.workgroupSize.x,
                  dispatch.workgroupSize.y,
                  dispatch.workgroupSize.z);
    dim3 blockCount(dispatch.workgroupCount.x,
                    dispatch.workgroupCount.y,
                    dispatch.workgroupCount.z);

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothVerticesCudaMain<<<blockCount, blockDim>>>();
    cudaCheckErrors("CUDA error during vertices smoothing");

    cudaDeviceSynchronize();
}
