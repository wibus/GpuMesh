#include "Base.cuh"

#include <iostream>

#include <DataStructures/NodeGroups.h>

#define POSITION_THREAD_COUNT uint(8)
#define ELEMENT_THREAD_COUNT uint(32)
#define ELEMENT_SLOT_COUNT uint(96)
#define SLOT_PER_THREAD (ELEMENT_SLOT_COUNT / ELEMENT_THREAD_COUNT)

#define GRAD_SAMP_COUNT uint(6)
#define LINE_SAMP_COUNT uint(8)

__constant__ int PGDSecurityCycleCount;
__constant__ float PGDLocalSizeToNodeShift;

struct PatchElem
{
    __device__ PatchElem() {}

    uint type;
    uint n;

    Tet tet;
    Pri pri;
    Hex hex;
    vec3 p[HEX_VERTEX_COUNT];
};


// Smoothing Helper
__device__ float computeLocalElementSize(uint vId);
__device__ void accumulatePatchQuality(
        double& patchQuality,
        double& patchWeight,
        double elemQuality);
__device__ float finalizePatchQuality(
        double patchQuality,
        double patchWeight);

__shared__ float nodeShift;
__shared__ float3 lineShift;
__shared__ extern PatchElem patchElems[];
__shared__ float elemQual[POSITION_THREAD_COUNT][ELEMENT_SLOT_COUNT];
__shared__ float patchQual[POSITION_THREAD_COUNT];


// ENTRY POINT //
__device__ void patchGradDsntSmoothVert(uint vId)
{
    const vec3 GRAD_SAMPS[GRAD_SAMP_COUNT] = {
        vec3(-1, 0, 0), vec3( 1, 0, 0), vec3(0, -1, 0),
        vec3(0,  1, 0), vec3(0, 0, -1), vec3(0, 0,  1)
    };

    const float LINE_SAMPS[LINE_SAMP_COUNT] = {
        -0.25, 0.00, 0.25, 0.50,
         0.75, 1.00, 1.25, 1.50
    };

    uint pId = threadIdx.x;
    uint eId = threadIdx.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint eBeg = (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint eEnd = ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;

    if(pId == 0)
    {
        for(uint e = eBeg; e < eEnd; ++e)
        {
            NeigElem elem = neigElems[topo.neigElemBase + e];
            patchElems[e].type = elem.type;
            patchElems[e].n = 0;

            switch(patchElems[e].type)
            {
            case TET_ELEMENT_TYPE :
                patchElems[e].tet = tets[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].tet.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].tet.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].tet.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].tet.v[3]].p;

                if(patchElems[e].tet.v[1] == vId) patchElems[e].n = 1;
                else if(patchElems[e].tet.v[2] == vId) patchElems[e].n = 2;
                else if(patchElems[e].tet.v[3] == vId) patchElems[e].n = 3;
                break;

            case PRI_ELEMENT_TYPE :
                patchElems[e].pri = pris[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].pri.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].pri.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].pri.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].pri.v[3]].p;
                patchElems[e].p[4] = verts[patchElems[e].pri.v[4]].p;
                patchElems[e].p[5] = verts[patchElems[e].pri.v[5]].p;

                if(patchElems[e].pri.v[1] == vId) patchElems[e].n = 1;
                else if(patchElems[e].pri.v[2] == vId) patchElems[e].n = 2;
                else if(patchElems[e].pri.v[3] == vId) patchElems[e].n = 3;
                else if(patchElems[e].pri.v[4] == vId) patchElems[e].n = 4;
                else if(patchElems[e].pri.v[5] == vId) patchElems[e].n = 5;
                break;

            case HEX_ELEMENT_TYPE :
                patchElems[e].hex = hexs[elem.id];
                patchElems[e].p[0] = verts[patchElems[e].hex.v[0]].p;
                patchElems[e].p[1] = verts[patchElems[e].hex.v[1]].p;
                patchElems[e].p[2] = verts[patchElems[e].hex.v[2]].p;
                patchElems[e].p[3] = verts[patchElems[e].hex.v[3]].p;
                patchElems[e].p[4] = verts[patchElems[e].hex.v[4]].p;
                patchElems[e].p[5] = verts[patchElems[e].hex.v[5]].p;
                patchElems[e].p[6] = verts[patchElems[e].hex.v[6]].p;
                patchElems[e].p[7] = verts[patchElems[e].hex.v[7]].p;

                if(patchElems[e].hex.v[1] == vId) patchElems[e].n = 1;
                else if(patchElems[e].hex.v[2] == vId) patchElems[e].n = 2;
                else if(patchElems[e].hex.v[3] == vId) patchElems[e].n = 3;
                else if(patchElems[e].hex.v[4] == vId) patchElems[e].n = 4;
                else if(patchElems[e].hex.v[5] == vId) patchElems[e].n = 5;
                else if(patchElems[e].hex.v[6] == vId) patchElems[e].n = 6;
                else if(patchElems[e].hex.v[7] == vId) patchElems[e].n = 7;
                break;
            }
        }
    }

    if(pId == 0 && eId == 0)
    {
        // Compute local element size
        float localSize = computeLocalElementSize(vId);

        // Initialize node shift distance
        nodeShift = localSize * PGDLocalSizeToNodeShift;
    }

    __syncthreads();

    float originalNodeShift = nodeShift;
    for(int c=0; c < PGDSecurityCycleCount; ++c)
    {
        vec3 pos = verts[vId].p;

        for(uint e = eBeg; e < eEnd; ++e)
        {
            vec3 vertPos[HEX_VERTEX_COUNT] = {
                patchElems[e].p[0],
                patchElems[e].p[1],
                patchElems[e].p[2],
                patchElems[e].p[3],
                patchElems[e].p[4],
                patchElems[e].p[5],
                patchElems[e].p[6],
                patchElems[e].p[7]
            };

            vertPos[patchElems[e].n] = pos + vec3(GRAD_SAMPS[pId]) * nodeShift;

            switch(patchElems[e].type)
            {
            case TET_ELEMENT_TYPE :
                elemQual[pId][e] = (*tetQualityImpl)(vertPos, patchElems[e].tet);
                break;
            case PRI_ELEMENT_TYPE :
                elemQual[pId][e] = (*priQualityImpl)(vertPos, patchElems[e].pri);
                break;
            case HEX_ELEMENT_TYPE :
                elemQual[pId][e] = (*hexQualityImpl)(vertPos, patchElems[e].hex);
                break;
            }
        }

        __syncthreads();

        if(eId == 0 && pId < GRAD_SAMP_COUNT)
        {
            double patchWeight = 0.0;
            double patchQuality = 0.0;
            for(uint e = 0; e < neigElemCount; ++e)
                accumulatePatchQuality(
                    patchQuality, patchWeight,
                    double(elemQual[pId][e]));

            patchQual[pId] = finalizePatchQuality(patchQuality, patchWeight);
        }

        __syncthreads();

        if(eId == 0 && pId == 0)
        {
            vec3 gradQ = vec3(
                patchQual[1] - patchQual[0],
                patchQual[3] - patchQual[2],
                patchQual[5] - patchQual[4]);
            float gradQNorm = length(gradQ);

            if(gradQNorm != 0)
            {
                lineShift = toFloat3(gradQ * (nodeShift / gradQNorm));
            }
            else
            {
                lineShift = make_float3(0, 0, 0);
            }
        }

        __syncthreads();

        if(lineShift.x == 0 && lineShift.y == 0 && lineShift.z == 0)
            break;

        for(uint e = eBeg; e < eEnd; ++e)
        {
            vec3 vertPos[HEX_VERTEX_COUNT] = {
                patchElems[e].p[0],
                patchElems[e].p[1],
                patchElems[e].p[2],
                patchElems[e].p[3],
                patchElems[e].p[4],
                patchElems[e].p[5],
                patchElems[e].p[6],
                patchElems[e].p[7]
            };

            vertPos[patchElems[e].n] = pos + toVec3(lineShift) * LINE_SAMPS[pId];
            switch(patchElems[e].type)
            {
            case TET_ELEMENT_TYPE :
                elemQual[pId][e] = (*tetQualityImpl)(vertPos, patchElems[e].tet);
                break;
            case PRI_ELEMENT_TYPE :
                elemQual[pId][e] = (*priQualityImpl)(vertPos, patchElems[e].pri);
                break;
            case HEX_ELEMENT_TYPE :
                elemQual[pId][e] = (*hexQualityImpl)(vertPos, patchElems[e].hex);
                break;
            }
        }

        __syncthreads();

        if(eId == 0)
        {
            double patchWeight = 0.0;
            double patchQuality = 0.0;
            for(uint e = 0; e < neigElemCount; ++e)
                accumulatePatchQuality(
                    patchQuality, patchWeight,
                    double(elemQual[pId][e]));

            patchQual[pId] = finalizePatchQuality(patchQuality, patchWeight);
        }

        __syncthreads();

        if(eId == 0 && pId == 0)
        {
            uint bestProposition = 0;
            float bestQualityMean = patchQual[0];
            for(uint p=1; p < LINE_SAMP_COUNT; ++p)
            {
                if(patchQual[p] > bestQualityMean)
                {
                    bestQualityMean = patchQual[p];
                    bestProposition = p;
                }
            }

            // Update vertex's position
            verts[vId].p = pos + toVec3(lineShift) * LINE_SAMPS[bestProposition];

            // Scale node shift and stop if it is too small
            nodeShift *= abs(LINE_SAMPS[bestProposition]);
        }

        __syncthreads();

        if(nodeShift < originalNodeShift / 10.0)
            break;
    }
}

__global__ void smoothPatchGradDsntVerticesCudaMain()
{
    if(blockIdx.x < GroupSize)
    {
        uint idx = GroupBase + blockIdx.x;
        uint vId = groupMembers[idx];
        patchGradDsntSmoothVert(vId);
    }
}

//__device__ smoothVertFct patchGradDsntSmoothVertPtr = patchGradDsntSmoothVert;



// CUDA Drivers
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch);

void installCudaPatchGradDsntSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift)
{
    // Main function is directly calling patchGradDsntSmoothVert

//    smoothVertFct d_smoothVert = nullptr;
//    cudaMemcpyFromSymbol(&d_smoothVert, patchGradDsntSmoothVertPtr, sizeof(smoothVertFct));
//    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));

    cudaMemcpyToSymbol(PGDSecurityCycleCount, &h_securityCycleCount, sizeof(int));
    cudaMemcpyToSymbol(PGDLocalSizeToNodeShift, &h_localSizeToNodeShift, sizeof(float));


    if(verboseCuda)
        printf("I -> CUDA \tPatch Gradient Decsent smoother installed\n");
}

void smoothCudaPatchGradDsntVertices(
        const NodeGroups::GpuDispatch& dispatch)
{
    setupCudaIndependentDispatch(dispatch);

    dim3 blockDim(POSITION_THREAD_COUNT, ELEMENT_THREAD_COUNT);
    size_t sharedDim = sizeof(PatchElem) * ELEMENT_SLOT_COUNT;
    //std::cout << "Requested shared memory size: " << sharedDim/1000.0 << "kB" << std::endl;

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothPatchGradDsntVerticesCudaMain<<<dispatch.gpuBufferSize, blockDim, sharedDim>>>();
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA error during vertices smoothing");
}
