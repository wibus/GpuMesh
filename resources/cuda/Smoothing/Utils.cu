#include "../Mesh.cuh"
#include <Smoothers/AbstractSmoother.h>


// Independent group range
__device__ int GroupBase;
__device__ int GroupSize;


// Invocation indices
__device__ uint getInvocationVertexId()
{
    // Default value is invalid
    // See isSmoothableVertex()
    uint vId = verts_length;

    // Assign real index only if this
    // invocation does not overflow
    uint gIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if(gIdx < GroupSize)
        vId = groupMembers[GroupBase + gIdx];

    return vId;
}

__device__ uint getInvocationTetId()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ uint getInvocationPriId()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ uint getInvocationHexId()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}


// Entity's smoothability
__device__ bool isSmoothableVertex(uint vId)
{
    if(vId >= verts_length)
        return false;

    Topo topo = topos[vId];
    if(topo.type == TOPO_FIXED)
        return false;

    if(topo.neigElemCount == 0)
        return false;

    return true;
}

__device__ bool isSmoothableTet(uint eId)
{
    if(eId >= tets_length)
        return false;

    return true;
}

__device__ bool isSmoothablePri(uint eId)
{
    if(eId >= pris_length)
        return false;

    return true;
}

__device__ bool isSmoothableHex(uint eId)
{
    if(eId >= hexs_length)
        return false;

    return true;
}


// CUDA Drivers
void setupCudaIndependentDispatch(const IndependentDispatch& dispatch)
{
    cudaMemcpyToSymbol(GroupBase, &dispatch.base, sizeof(int));
    cudaMemcpyToSymbol(GroupSize, &dispatch.size, sizeof(int));
}
