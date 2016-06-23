#include "Base.cuh"


__device__ smoothTetFct smoothTet = nullptr;
__device__ smoothPriFct smoothPri = nullptr;
__device__ smoothHexFct smoothHex = nullptr;


// Smoothing helper
__device__ uint getInvocationTetId();
__device__ uint getInvocationPriId();
__device__ uint getInvocationHexId();
__device__ bool isSmoothableTet(uint eId);
__device__ bool isSmoothablePri(uint eId);
__device__ bool isSmoothableHex(uint eId);


__global__ void smoothElementsCudaMain()
{
    uint tetId = getInvocationTetId();
    if(isSmoothableTet(tetId))
    {
        smoothTet(tetId);
    }

    uint priId = getInvocationPriId();
    if(isSmoothablePri(priId))
    {
        smoothPri(priId);
    }

    uint hexId = getInvocationHexId();
    if(isSmoothableHex(hexId))
    {
        smoothHex(hexId);
    }
}


// CUDA Drivers
void smoothCudaElements(size_t workGroupCount, size_t workgroupSize)
{
    cudaCheckErrors("CUDA error before elements smoothing");
    smoothElementsCudaMain<<<workGroupCount, workgroupSize>>>();
    cudaCheckErrors("CUDA error during elements smoothing");

    cudaDeviceSynchronize();
}
