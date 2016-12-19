#include "Base.cuh"

#include <iostream>

#include <DataStructures/NodeGroups.h>

#define NODE_THREAD_COUNT uint(8)
#define ELEMENT_THREAD_COUNT uint(32)
#define ELEMENT_PER_THREAD_COUNT uint(3)
#define POSITION_SLOT_COUNT uint(8)

#define GRAD_SAMP_COUNT uint(6)
#define LINE_SAMP_COUNT uint(8)

#define MIN_MAX 2147483647


namespace megd
{
    __constant__ int SECURITY_CYCLE_COUNT;
    __constant__ float LOCAL_SIZE_TO_NODE_SHIFT;

    __shared__ float nodeShift[NODE_THREAD_COUNT];
    __shared__ int patchMin[NODE_THREAD_COUNT][POSITION_SLOT_COUNT];
    __shared__ float patchMean[NODE_THREAD_COUNT][POSITION_SLOT_COUNT];
    __shared__ float patchQual[NODE_THREAD_COUNT][POSITION_SLOT_COUNT];
}

using namespace megd;


// Smoothing Helper
__device__ uint getInvocationVertexId();
__device__ bool isSmoothableVertex(uint vId);
__device__ float computeLocalElementSize(uint vId);


// ENTRY POINT //
__device__ void multiElemGradDsntSmoothVert(uint vId)
{
    const vec3 GRAD_SAMPS[GRAD_SAMP_COUNT] = {
        vec3(-1, 0, 0), vec3( 1, 0, 0), vec3(0, -1, 0),
        vec3(0,  1, 0), vec3(0, 0, -1), vec3(0, 0,  1)
    };

    const float LINE_SAMPS[LINE_SAMP_COUNT] = {
        -0.25, 0.00, 0.25, 0.50,
         0.75, 1.00, 1.25, 1.50
    };

    uint nId = threadIdx.x;
    uint eId = threadIdx.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint eBeg = (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint eEnd = ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint nBeg = topo.neigElemBase + eBeg;
    uint nEnd = topo.neigElemBase + eEnd;

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

    if(eId < POSITION_SLOT_COUNT)
    {
        patchMin[nId][eId] = MIN_MAX;
        patchMean[nId][eId] = 0.0;
    }

    if(eId == 0)
    {
        // Compute local element size
        float localSize = computeLocalElementSize(vId);

        // Initialize node shift distance
        nodeShift[nId] = localSize * LOCAL_SIZE_TO_NODE_SHIFT;
    }

    __syncthreads();


    float originalNodeShift = nodeShift[nId];
    for(int c=0; c < SECURITY_CYCLE_COUNT; ++c)
    {
        vec3 pos = verts[vId].p;

        for(uint p=0; p < GRAD_SAMP_COUNT; ++p)
        {
            vec3 gradSamp = pos + GRAD_SAMPS[p] * nodeShift[nId];

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

                atomicMin(&patchMin[nId][p], qual * MIN_MAX);
                atomicAdd(&patchMean[nId][p], 1.0 / qual);
            }
        }

        __syncthreads();


        if(eId < GRAD_SAMP_COUNT)
        {
            if(patchMin[nId][eId] <= 0.0)
                patchQual[nId][eId] = patchMin[nId][eId] / float(MIN_MAX);
            else
                patchQual[nId][eId] = neigElemCount / patchMean[nId][eId];

            patchMin[nId][eId] = MIN_MAX;
            patchMean[nId][eId] = 0.0;
        }

        __syncthreads();


        vec3 gradQ = vec3(
            patchQual[nId][1] - patchQual[nId][0],
            patchQual[nId][3] - patchQual[nId][2],
            patchQual[nId][5] - patchQual[nId][4]);
        float gradQNorm = length(gradQ);

        vec3 lineShift;
        if(gradQNorm != 0)
            lineShift = gradQ * (nodeShift[nId] / gradQNorm);
        else
            break;


        for(uint p=0; p < LINE_SAMP_COUNT; ++p)
        {
            vec3 lineSamp = pos + lineShift * LINE_SAMPS[p];

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

                atomicMin(&patchMin[nId][p], qual * MIN_MAX);
                atomicAdd(&patchMean[nId][p], 1.0 / qual);
            }
        }

        __syncthreads();


        if(eId < LINE_SAMP_COUNT)
        {
            if(patchMin[nId][eId] <= 0.0)
                patchQual[nId][eId] = patchMin[nId][eId] / float(MIN_MAX);
            else
                patchQual[nId][eId] = neigElemCount / patchMean[nId][eId];

            patchMin[nId][eId] = MIN_MAX;
            patchMean[nId][eId] = 0.0;
        }

        __syncthreads();


        if(eId == 0)
        {
            uint bestProposition = 0;
            float bestQualityMean = patchQual[nId][0];
            for(uint p=1; p < LINE_SAMP_COUNT; ++p)
            {
                if(patchQual[nId][p] > bestQualityMean)
                {
                    bestQualityMean = patchQual[nId][p];
                    bestProposition = p;
                }
            }

            // Update vertex's position
            verts[vId].p = pos + lineShift * LINE_SAMPS[bestProposition];

            // Scale node shift and stop if it is too small
            nodeShift[nId] *= abs(LINE_SAMPS[bestProposition]);
        }

        __syncthreads();


        if(nodeShift[nId] < originalNodeShift / 10.0)
            break;
    }
}

__global__ void smoothMultiElemGradDsntVerticesCudaMain()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        multiElemGradDsntSmoothVert(vId);
    }
}

//__device__ smoothVertFct patchGradDsntSmoothVertPtr = patchGradDsntSmoothVert;



// CUDA Drivers
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch);

void installCudaMultiElemGradDsntSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift)
{
    // Main function is directly calling patchGradDsntSmoothVert

//    smoothVertFct d_smoothVert = nullptr;
//    cudaMemcpyFromSymbol(&d_smoothVert, patchGradDsntSmoothVertPtr, sizeof(smoothVertFct));
//    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));

    cudaMemcpyToSymbol(SECURITY_CYCLE_COUNT, &h_securityCycleCount, sizeof(int));
    cudaMemcpyToSymbol(LOCAL_SIZE_TO_NODE_SHIFT, &h_localSizeToNodeShift, sizeof(float));

    cudaFuncSetCacheConfig(smoothMultiElemGradDsntVerticesCudaMain, cudaFuncCachePreferL1);

    if(verboseCuda)
        printf("I -> CUDA \tPatch Gradient Decsent smoother installed\n");
}

void smoothCudaMultiElemGradDsntVertices(
        const NodeGroups::GpuDispatch& dispatch)
{
    setupCudaIndependentDispatch(dispatch);

    assert(ELEMENT_THREAD_COUNT >= POSITION_SLOT_COUNT);

    dim3 blockDim(NODE_THREAD_COUNT, ELEMENT_THREAD_COUNT);

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothMultiElemGradDsntVerticesCudaMain<<<dispatch.gpuBufferSize, blockDim>>>();
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA error during vertices smoothing");
}
