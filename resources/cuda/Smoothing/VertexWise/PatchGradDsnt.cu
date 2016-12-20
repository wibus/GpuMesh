#include "Base.cuh"

#include <iostream>

#include <DataStructures/NodeGroups.h>

#define POSITION_THREAD_COUNT uint(8)
#define ELEMENT_THREAD_COUNT uint(8)

#define ELEMENT_PER_THREAD_COUNT uint(96 / ELEMENT_THREAD_COUNT)

#define GRAD_SAMP_COUNT uint(6)
#define LINE_SAMP_COUNT uint(8)

#define MIN_MAX 2147483647


namespace pgd
{
    __constant__ int SECURITY_CYCLE_COUNT;
    __constant__ float LOCAL_SIZE_TO_NODE_SHIFT;

    __shared__ float nodeShift;
    __shared__ int patchMin[POSITION_THREAD_COUNT];
    __shared__ float patchMean[POSITION_THREAD_COUNT];
    __shared__ float patchQual[POSITION_THREAD_COUNT];
}

using namespace pgd;


// Smoothing Helper
__device__ float computeLocalElementSize(uint vId);


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
    uint neigBase = topo.neigElemBase;
    uint neigElemCount = topo.neigElemCount;
    uint eBeg = (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint eEnd = ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint nBeg = neigBase + eBeg;
    uint nEnd = neigBase + eEnd;

    PatchElem elems[ELEMENT_PER_THREAD_COUNT];
    for(uint e=0, ne = nBeg; ne < nEnd; ++e, ++ne)
    {
        NeigElem elem = neigElems[ne];
        elems[e].type = elem.type;
        elems[e].n = elem.vId;

        switch(elems[e].type)
        {
        case TET_ELEMENT_TYPE :
            elems[e].tet = tets[elem.id];
            elems[e].p[0] = verts[elems[e].tet.v[0]].p;
            elems[e].p[1] = verts[elems[e].tet.v[1]].p;
            elems[e].p[2] = verts[elems[e].tet.v[2]].p;
            elems[e].p[3] = verts[elems[e].tet.v[3]].p;
            break;

        case PRI_ELEMENT_TYPE :
            elems[e].pri = pris[elem.id];
            elems[e].p[0] = verts[elems[e].pri.v[0]].p;
            elems[e].p[1] = verts[elems[e].pri.v[1]].p;
            elems[e].p[2] = verts[elems[e].pri.v[2]].p;
            elems[e].p[3] = verts[elems[e].pri.v[3]].p;
            elems[e].p[4] = verts[elems[e].pri.v[4]].p;
            elems[e].p[5] = verts[elems[e].pri.v[5]].p;
            break;

        case HEX_ELEMENT_TYPE :
            elems[e].hex = hexs[elem.id];
            elems[e].p[0] = verts[elems[e].hex.v[0]].p;
            elems[e].p[1] = verts[elems[e].hex.v[1]].p;
            elems[e].p[2] = verts[elems[e].hex.v[2]].p;
            elems[e].p[3] = verts[elems[e].hex.v[3]].p;
            elems[e].p[4] = verts[elems[e].hex.v[4]].p;
            elems[e].p[5] = verts[elems[e].hex.v[5]].p;
            elems[e].p[6] = verts[elems[e].hex.v[6]].p;
            elems[e].p[7] = verts[elems[e].hex.v[7]].p;
            break;
        }
    }

    if(eId == 0)
    {
        patchMin[pId] = MIN_MAX;
        patchMean[pId] = 0.0;
    }

    if(pId == 0 && eId == 0)
    {
        // Compute local element size
        float localSize = computeLocalElementSize(vId);

        // Initialize node shift distance
        nodeShift = localSize * LOCAL_SIZE_TO_NODE_SHIFT;
    }

    __syncthreads();


    float originalNodeShift = nodeShift;
    for(int c=0; c < SECURITY_CYCLE_COUNT; ++c)
    {
        vec3 pos = verts[vId].p;

        if(pId < GRAD_SAMP_COUNT)
        {
            vec3 gradSamp = pos + GRAD_SAMPS[pId] * nodeShift;

            for(uint e=0, id = eBeg; id < eEnd; ++e, ++id)
            {
                elems[e].p[elems[e].n] = gradSamp;

                float qual = 0.0;
                switch(elems[e].type)
                {
                case TET_ELEMENT_TYPE :
                    qual = (*tetQualityImpl)(elems[e].p, elems[e].tet);
                    break;
                case PRI_ELEMENT_TYPE :
                    qual = (*priQualityImpl)(elems[e].p, elems[e].pri);
                    break;
                case HEX_ELEMENT_TYPE :
                    qual = (*hexQualityImpl)(elems[e].p, elems[e].hex);
                    break;
                }

                atomicMin(&patchMin[pId], qual * MIN_MAX);
                atomicAdd(&patchMean[pId], 1.0 / qual);
            }
        }

        __syncthreads();


        if(eId == 0)
        {
            if(patchMin[pId] <= 0.0)
                patchQual[pId] = patchMin[pId] / float(MIN_MAX);
            else
                patchQual[pId] = neigElemCount / patchMean[pId];

            patchMin[pId] = MIN_MAX;
            patchMean[pId] = 0.0;
        }

        __syncthreads();


        vec3 gradQ = vec3(
            patchQual[1] - patchQual[0],
            patchQual[3] - patchQual[2],
            patchQual[5] - patchQual[4]);
        float gradQNorm = length(gradQ);

        vec3 lineShift;
        if(gradQNorm != 0)
            lineShift = gradQ * (nodeShift / gradQNorm);
        else
            break;


        vec3 lineSamp = pos + lineShift * LINE_SAMPS[pId];

        for(uint e=0, id = eBeg; id < eEnd; ++e, ++id)
        {
            elems[e].p[elems[e].n] = lineSamp;

            float qual = 0.0;
            switch(elems[e].type)
            {
            case TET_ELEMENT_TYPE :
                qual = (*tetQualityImpl)(elems[e].p, elems[e].tet);
                break;
            case PRI_ELEMENT_TYPE :
                qual = (*priQualityImpl)(elems[e].p, elems[e].pri);
                break;
            case HEX_ELEMENT_TYPE :
                qual = (*hexQualityImpl)(elems[e].p, elems[e].hex);
                break;
            }

            atomicMin(&patchMin[pId], qual * MIN_MAX);
            atomicAdd(&patchMean[pId], 1.0 / qual);
        }

        __syncthreads();


        if(eId == 0)
        {
            if(patchMin[pId] <= 0.0)
                patchQual[pId] = patchMin[pId] / float(MIN_MAX);
            else
                patchQual[pId] = neigElemCount / patchMean[pId];

            patchMin[pId] = MIN_MAX;
            patchMean[pId] = 0.0;
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
            verts[vId].p = pos + lineShift * LINE_SAMPS[bestProposition];

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

    cudaMemcpyToSymbol(SECURITY_CYCLE_COUNT, &h_securityCycleCount, sizeof(int));
    cudaMemcpyToSymbol(LOCAL_SIZE_TO_NODE_SHIFT, &h_localSizeToNodeShift, sizeof(float));

    cudaFuncSetCacheConfig(smoothPatchGradDsntVerticesCudaMain, cudaFuncCachePreferL1);

    if(verboseCuda)
        printf("I -> CUDA \tPatch Gradient Decsent smoother installed\n");
}

void smoothCudaPatchGradDsntVertices(
        const NodeGroups::GpuDispatch& dispatch)
{
    setupCudaIndependentDispatch(dispatch);

    dim3 blockDim(POSITION_THREAD_COUNT, ELEMENT_THREAD_COUNT);

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothPatchGradDsntVerticesCudaMain<<<dispatch.workgroupCount, blockDim>>>();
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA error during vertices smoothing");
}
