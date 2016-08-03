#include "../Mesh.cuh"
#include <DataStructures/GpuMesh.h>
#include <DataStructures/NodeGroups.h>
#include <Smoothers/AbstractSmoother.h>


// Independent group range
__constant__ int GroupBase;
__constant__ int GroupSize;


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
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch)
{
    cudaMemcpyToSymbol(GroupBase, &dispatch.gpuBufferBase, sizeof(int));
    cudaMemcpyToSymbol(GroupSize, &dispatch.gpuBufferSize, sizeof(int));
}

void fetchCudaSubsurfaceVertices(
        std::vector<MeshVert>& meshVerts,
        const NodeGroups::ParallelGroup& group)
{
    size_t subsurfaceCount =
            group.subsurfaceRange.end -
            group.subsurfaceRange.begin;

    if(subsurfaceCount > 0)
    {
        size_t subsurfaceSize = subsurfaceCount * sizeof(GpuVert);
        size_t subsurfaceBase = group.subsurfaceRange.begin;

        GpuVert* d_verts = nullptr;
        GpuVert* h_verts = new GpuVert[subsurfaceCount];
        cudaCheckErrors("CUDA error fetch subsurface");
        cudaMemcpyFromSymbol(&d_verts, verts, sizeof(d_verts));
        cudaCheckErrors("CUDA error fetch subsurface");
        cudaMemcpy(h_verts, d_verts + subsurfaceBase, subsurfaceSize, cudaMemcpyDeviceToHost);
        cudaCheckErrors("CUDA error fetch subsurface");

        for(size_t vId = group.subsurfaceRange.begin, bId=0;
            vId < group.subsurfaceRange.end; ++vId, ++bId)
        {
            MeshVert vert(h_verts[bId]);
            meshVerts[vId] = vert;
        }

        delete [] h_verts;
    }
}

void sendCudaBoundaryVertices(
        const std::vector<MeshVert>& meshVerts,
        const NodeGroups::ParallelGroup& group)
{
    size_t boundaryCount =
            group.boundaryRange.end -
            group.boundaryRange.begin;

    if(boundaryCount > 0)
    {
        size_t boundarySize = boundaryCount * sizeof(GpuVert);
        size_t boundaryBase = group.boundaryRange.begin;

        GpuVert* h_verts = new GpuVert[boundaryCount];
        for(size_t vId = group.boundaryRange.begin, bId=0;
            vId < group.boundaryRange.end; ++vId, ++bId)
        {
            GpuVert vert(meshVerts[vId]);
            h_verts[bId] = vert;
        }

        GpuVert* d_verts = nullptr;
        cudaCheckErrors("CUDA error send boundary");
        cudaMemcpyFromSymbol(&d_verts, verts, sizeof(d_verts));
        cudaCheckErrors("CUDA error send boundary");
        cudaMemcpy(d_verts + boundaryBase, h_verts, boundarySize, cudaMemcpyHostToDevice);
        cudaCheckErrors("CUDA error send boundary");

        delete [] h_verts;
    }
}
