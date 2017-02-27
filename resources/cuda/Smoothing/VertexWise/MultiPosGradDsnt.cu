#include "Base.cuh"

#include <iostream>

#include <DataStructures/NodeGroups.h>

#define POSITION_THREAD_COUNT uint(8)
#define NODE_THREAD_COUNT uint(4)

#define GRAD_SAMP_COUNT uint(6)
#define LINE_SAMP_COUNT uint(8)


namespace mpgd
{
    __constant__ int SECURITY_CYCLE_COUNT;
    __constant__ float LOCAL_SIZE_TO_NODE_SHIFT;

    __shared__ float nodeShift[NODE_THREAD_COUNT];
    __shared__ float patchQual[NODE_THREAD_COUNT][POSITION_THREAD_COUNT];
}

using namespace mpgd;


// Smoothing Helper
__device__ float computeLocalElementSize(uint vId);


// ENTRY POINT //
__device__ void multiPosGradDsntSmoothVert(uint vId)
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
    uint nId = threadIdx.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint eBeg = topo.neigElemBase;
    uint eEnd = topo.neigElemBase + neigElemCount;

    if(pId == 0)
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

        float patchMin = 1.0;
        double patchMean = 0.0;

        if(pId < GRAD_SAMP_COUNT)
        {
            vec3 gradSamp = pos + GRAD_SAMPS[pId] * nodeShift[nId];

            for(uint e = eBeg; e < eEnd; ++e)
            {
                NeigElem& elem = neigElems[e];
                vec3 vertPos[HEX_VERTEX_COUNT];

                float qual = 0.0;
                switch(elem.type)
                {
                case TET_ELEMENT_TYPE :
                    vertPos[0] = verts[tets[elem.id].v[0]].p;
                    vertPos[1] = verts[tets[elem.id].v[1]].p;
                    vertPos[2] = verts[tets[elem.id].v[2]].p;
                    vertPos[3] = verts[tets[elem.id].v[3]].p;
                    vertPos[elem.vId] = gradSamp;
                    qual = (*tetQualityImpl)(vertPos, tets[elem.id]);
                    break;

                case PRI_ELEMENT_TYPE :
                    vertPos[0] = verts[pris[elem.id].v[0]].p;
                    vertPos[1] = verts[pris[elem.id].v[1]].p;
                    vertPos[2] = verts[pris[elem.id].v[2]].p;
                    vertPos[3] = verts[pris[elem.id].v[3]].p;
                    vertPos[4] = verts[pris[elem.id].v[4]].p;
                    vertPos[5] = verts[pris[elem.id].v[5]].p;
                    vertPos[elem.vId] = gradSamp;
                    qual = (*priQualityImpl)(vertPos, pris[elem.id]);
                    break;

                case HEX_ELEMENT_TYPE :
                    vertPos[0] = verts[hexs[elem.id].v[0]].p;
                    vertPos[1] = verts[hexs[elem.id].v[1]].p;
                    vertPos[2] = verts[hexs[elem.id].v[2]].p;
                    vertPos[3] = verts[hexs[elem.id].v[3]].p;
                    vertPos[4] = verts[hexs[elem.id].v[4]].p;
                    vertPos[5] = verts[hexs[elem.id].v[5]].p;
                    vertPos[6] = verts[hexs[elem.id].v[6]].p;
                    vertPos[7] = verts[hexs[elem.id].v[7]].p;
                    vertPos[elem.vId] = gradSamp;
                    qual = (*hexQualityImpl)(vertPos, hexs[elem.id]);
                    break;
                }

                patchMin = min(patchMin, qual);
                patchMean += 1.0 / qual;
            }

            if(patchMin <= 0.0)
                patchQual[nId][pId] = patchMin;
            else
                patchQual[nId][pId] = neigElemCount / patchMean;
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


        patchMin = 1.0;
        patchMean = 0.0;

        vec3 lineSamp = pos + lineShift * LINE_SAMPS[pId];

        for(uint e = eBeg; e < eEnd; ++e)
        {
            NeigElem& elem = neigElems[e];
            vec3 vertPos[HEX_VERTEX_COUNT];

            float qual = 0.0;
            switch(elem.type)
            {
            case TET_ELEMENT_TYPE :
                vertPos[0] = verts[tets[elem.id].v[0]].p;
                vertPos[1] = verts[tets[elem.id].v[1]].p;
                vertPos[2] = verts[tets[elem.id].v[2]].p;
                vertPos[3] = verts[tets[elem.id].v[3]].p;
                vertPos[elem.vId] = lineSamp;
                qual = (*tetQualityImpl)(vertPos, tets[elem.id]);
                break;

            case PRI_ELEMENT_TYPE :
                vertPos[0] = verts[pris[elem.id].v[0]].p;
                vertPos[1] = verts[pris[elem.id].v[1]].p;
                vertPos[2] = verts[pris[elem.id].v[2]].p;
                vertPos[3] = verts[pris[elem.id].v[3]].p;
                vertPos[4] = verts[pris[elem.id].v[4]].p;
                vertPos[5] = verts[pris[elem.id].v[5]].p;
                vertPos[elem.vId] = lineSamp;
                qual = (*priQualityImpl)(vertPos, pris[elem.id]);
                break;

            case HEX_ELEMENT_TYPE :
                vertPos[0] = verts[hexs[elem.id].v[0]].p;
                vertPos[1] = verts[hexs[elem.id].v[1]].p;
                vertPos[2] = verts[hexs[elem.id].v[2]].p;
                vertPos[3] = verts[hexs[elem.id].v[3]].p;
                vertPos[4] = verts[hexs[elem.id].v[4]].p;
                vertPos[5] = verts[hexs[elem.id].v[5]].p;
                vertPos[6] = verts[hexs[elem.id].v[6]].p;
                vertPos[7] = verts[hexs[elem.id].v[7]].p;
                vertPos[elem.vId] = lineSamp;
                qual = (*hexQualityImpl)(vertPos, hexs[elem.id]);
                break;
            }

            patchMin = min(patchMin, qual);
            patchMean += 1.0 / qual;
        }

        if(patchMin <= 0.0)
            patchQual[nId][pId] = patchMin;
        else
            patchQual[nId][pId] = neigElemCount / patchMean;

        __syncthreads();


        if(pId == 0)
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

__global__ void smoothMultiPosGradDsntVerticesCudaMain()
{
    uint localId = blockIdx.x * blockDim.y + threadIdx.y;

    if(localId < GroupSize)
    {
        uint idx = GroupBase + localId;
        uint vId = groupMembers[idx];
        multiPosGradDsntSmoothVert(vId);
    }
}

//__device__ smoothVertFct patchGradDsntSmoothVertPtr = patchGradDsntSmoothVert;



// CUDA Drivers
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch);

void installCudaMultiPosGradDsntSmoother(
        int h_securityCycleCount,
        float h_localSizeToNodeShift)
{
    // Main function is directly calling patchGradDsntSmoothVert

//    smoothVertFct d_smoothVert = nullptr;
//    cudaMemcpyFromSymbol(&d_smoothVert, patchGradDsntSmoothVertPtr, sizeof(smoothVertFct));
//    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));

    cudaMemcpyToSymbol(SECURITY_CYCLE_COUNT, &h_securityCycleCount, sizeof(int));
    cudaMemcpyToSymbol(LOCAL_SIZE_TO_NODE_SHIFT, &h_localSizeToNodeShift, sizeof(float));

    cudaFuncSetCacheConfig(smoothMultiPosGradDsntVerticesCudaMain, cudaFuncCachePreferL1);

    if(verboseCuda)
        printf("I -> CUDA \tMulti Position Gradient Decsent smoother installed\n");
}

void smoothCudaMultiPosGradDsntVertices(
        const NodeGroups::GpuDispatch& dispatch)
{
    setupCudaIndependentDispatch(dispatch);

    dim3 blockDim(dispatch.workgroupSize.x,
                  dispatch.workgroupSize.y,
                  dispatch.workgroupSize.z);
    dim3 blockCount(dispatch.workgroupCount.x,
                    dispatch.workgroupCount.y,
                    dispatch.workgroupCount.z);

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothMultiPosGradDsntVerticesCudaMain<<<blockCount, blockDim>>>();
    cudaCheckErrors("CUDA error during vertices smoothing");

    cudaDeviceSynchronize();
}
