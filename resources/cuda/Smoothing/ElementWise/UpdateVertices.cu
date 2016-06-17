#include "Base.cuh"
#include <DataStructures/NodeGroups.h>
#include <Smoothers/AbstractSmoother.h>


// Vertex Accum
__device__ bool assignAverage(uint vId, vec3& pos);
__device__ void reinitAccum(uint vId);

// Smoothing Helper
__device__ uint getInvocationVertexId();
__device__ bool isSmoothableVertex(uint vId);
__device__ float patchQuality(uint vId);


__global__ void updateVerticesCudaMain()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        vec3 pos = verts[vId].p;
        vec3 posPrim = pos;

        if(assignAverage(vId, posPrim))
        {
            float prePatchQuality =
                patchQuality(vId);

            verts[vId].p = posPrim;

            float patchQualityPrime =
                patchQuality(vId);

            if(patchQualityPrime < prePatchQuality)
                verts[vId].p = pos;
        }

        reinitAccum(vId);
    }
}


// CUDA Drivers
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch);

void updateCudaSmoothedElementsVertices(
        const NodeGroups::GpuDispatch& dispatch,
        size_t workgroupSize)
{
    setupCudaIndependentDispatch(dispatch);

    cudaCheckErrors("CUDA error before vertices update");
    updateVerticesCudaMain<<<dispatch.workgroupCount, workgroupSize>>>();
    cudaCheckErrors("CUDA error in vertices update");

    cudaDeviceSynchronize();
}
