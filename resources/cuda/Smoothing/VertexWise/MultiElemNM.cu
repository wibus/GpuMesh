#include "Base.cuh"

#include <DataStructures/NodeGroups.h>


#define NODE_THREAD_COUNT uint(4)
#define ELEMENT_THREAD_COUNT uint(8)

#define ELEMENT_PER_THREAD_COUNT uint(96 / ELEMENT_THREAD_COUNT)

#define MIN_MAX 2147483647


namespace menm
{
    __constant__ float VALUE_CONVERGENCE;
    __constant__ int SECURITY_CYCLE_COUNT;
    __constant__ float LOCALE_SIZE_TO_NODE_SHIFT;
    __constant__ float ALPHA;
    __constant__ float BETA;
    __constant__ float GAMMA;
    __constant__ float DELTA;

    __shared__ int patchMin[NODE_THREAD_COUNT];
    __shared__ float patchMean[NODE_THREAD_COUNT];
}

using namespace menm;


// Smoothing Helper
__device__ uint getInvocationVertexId();
__device__ bool isSmoothableVertex(uint vId);
__device__ float computeLocalElementSize(uint vId);

__device__ float multiElemPatchQuality(
        uint nBeg, uint nEnd,
        uint neigElemCount,
        const vec3& pos)
{
    uint nId = threadIdx.x;

    patchMin[nId] = MIN_MAX;
    patchMean[nId] = 0.0;

    __syncthreads();


    for(uint e = nBeg; e < nEnd; ++e)
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
            vertPos[elem.vId] = pos;
            qual = (*tetQualityImpl)(vertPos, tets[elem.id]);
            break;

        case PRI_ELEMENT_TYPE :
            vertPos[0] = verts[pris[elem.id].v[0]].p;
            vertPos[1] = verts[pris[elem.id].v[1]].p;
            vertPos[2] = verts[pris[elem.id].v[2]].p;
            vertPos[3] = verts[pris[elem.id].v[3]].p;
            vertPos[4] = verts[pris[elem.id].v[4]].p;
            vertPos[5] = verts[pris[elem.id].v[5]].p;
            vertPos[elem.vId] = pos;
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
            vertPos[elem.vId] = pos;
            qual = (*hexQualityImpl)(vertPos, hexs[elem.id]);
            break;
        }

        atomicMin(&patchMin[nId], qual * MIN_MAX);
        atomicAdd(&patchMean[nId], 1.0 / qual);
    }

    __syncthreads();


    float patchQual = 0.0;

    if(patchMin[nId] <= 0.0)
        patchQual = patchMin[nId] / float(MIN_MAX);
    else
        patchQual = neigElemCount / patchMean[nId];

    return patchQual;
}


// ENTRY POINT //
__device__ void multiElemNMSmoothVert(uint vId)
{
    uint eId = threadIdx.y;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;
    uint nBeg = topo.neigElemBase + (eId * neigElemCount) / ELEMENT_THREAD_COUNT;
    uint nEnd = topo.neigElemBase + ((eId+1) * neigElemCount) / ELEMENT_THREAD_COUNT;


    // Compute local element size
    float localSize = computeLocalElementSize(vId);

    // Initialize node shift distance
    float nodeShift = localSize * LOCALE_SIZE_TO_NODE_SHIFT;

    vec3 pos = verts[vId].p;
    vec4 vo(pos, multiElemPatchQuality(nBeg, nEnd, neigElemCount, pos));

    vec4 simplex[TET_VERTEX_COUNT] = {
        vec4(pos + vec3(nodeShift, 0, 0), 0),
        vec4(pos + vec3(0, nodeShift, 0), 0),
        vec4(pos + vec3(0, 0, nodeShift), 0),
        vo
    };

    int cycle = 0;
    bool reset = false;
    bool terminated = false;
    while(!terminated)
    {
        for(uint p=0; p < TET_VERTEX_COUNT-1; ++p)
        {
            // Since 'pos' is a reference on vertex's position
            // modifing its value here should be seen by the evaluator
            verts[vId].p = vec3(simplex[p]);

            // Compute patch quality
            simplex[p] = vec4(verts[vId].p, multiElemPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p));
        }

        // Mini bubble sort
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[2].w > simplex[3].w)
            swap(simplex[2], simplex[3]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);


        for(; cycle < SECURITY_CYCLE_COUNT; ++cycle)
        {
            // Centroid
            vec3 c = 1/3.0f * (
                vec3(simplex[1]) +
                vec3(simplex[2]) +
                vec3(simplex[3]));

            float f = 0.0;

            // Reflect
            verts[vId].p = c + ALPHA*(c - vec3(simplex[0]));
            float fr = f = multiElemPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);

            vec3 xr = verts[vId].p;

            // Expand
            if(simplex[3].w < fr)
            {
                verts[vId].p = c + GAMMA*(verts[vId].p - c);
                float fe = f = multiElemPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);

                if(fe <= fr)
                {
                    verts[vId].p = xr;
                    f = fr;
                }
            }
            // Contract
            else if(simplex[1].w >= fr)
            {
                // Outside
                if(fr > simplex[0].w)
                {
                    verts[vId].p = c + BETA*(vec3(xr) - c);
                    f = multiElemPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);
                }
                // Inside
                else
                {
                    verts[vId].p = c + BETA*(vec3(simplex[0]) - c), 0;
                    f = multiElemPatchQuality(nBeg, nEnd, neigElemCount, verts[vId].p);
                }
            }

            // Insert new vertex in the working simplex
            vec4 vertex(verts[vId].p, f);
            if(vertex.w > simplex[3].w)
                swap(simplex[3], vertex);
            if(vertex.w > simplex[2].w)
                swap(simplex[2], vertex);
            if(vertex.w > simplex[1].w)
                swap(simplex[1], vertex);
            if(vertex.w > simplex[0].w)
                swap(simplex[0], vertex);


            if( (simplex[3].w - simplex[1].w) < VALUE_CONVERGENCE )
            {
                terminated = true;
                break;
            }
        }

        if( terminated || (cycle >= SECURITY_CYCLE_COUNT && reset) )
        {
            break;
        }
        else
        {
            simplex[0] = vo - vec4(nodeShift, 0, 0, 0);
            simplex[1] = vo - vec4(0, nodeShift, 0, 0);
            simplex[2] = vo - vec4(0, 0, nodeShift, 0);
            simplex[3] = vo;
            reset = true;
            cycle = 0;
        }
    }

    verts[vId].p = vec3(simplex[3]);
}

__global__ void smoothMultiElemNMVerticesCudaMain()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        multiElemNMSmoothVert(vId);
    }
}

// __device__ smoothVertFct multiElemNMSmoothVertPtr = multiElemNMSmoothVert;


// CUDA Drivers
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch);

void installCudaMultiElemNMSmoother(
        float h_valueConvergence,
        int h_securityCycleCount,
        float h_localSizeToNodeShift,
        float h_alpha,
        float h_beta,
        float h_gamma,
        float h_delta)
{
//    smoothVertFct d_smoothVert = nullptr;
//    cudaMemcpyFromSymbol(&d_smoothVert, multiElemNMSmoothVertPtr, sizeof(smoothVertFct));
//    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));

    cudaMemcpyToSymbol(VALUE_CONVERGENCE, &h_valueConvergence, sizeof(float));
    cudaMemcpyToSymbol(SECURITY_CYCLE_COUNT, &h_securityCycleCount, sizeof(int));
    cudaMemcpyToSymbol(LOCALE_SIZE_TO_NODE_SHIFT, &h_localSizeToNodeShift, sizeof(float));
    cudaMemcpyToSymbol(ALPHA, &h_alpha, sizeof(float));
    cudaMemcpyToSymbol(BETA,  &h_beta,  sizeof(float));
    cudaMemcpyToSymbol(GAMMA, &h_gamma, sizeof(float));
    cudaMemcpyToSymbol(DELTA, &h_delta, sizeof(float));


    if(verboseCuda)
        printf("I -> CUDA \tMulti Elem Nelder Mead smoother installed\n");
}

void smoothCudaMultiElemNMVertices(
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
    smoothMultiElemNMVerticesCudaMain<<<blockCount, blockDim>>>();
    cudaCheckErrors("CUDA error during vertices smoothing");

    cudaDeviceSynchronize();
}
