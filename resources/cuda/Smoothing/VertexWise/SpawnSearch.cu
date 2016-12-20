#include "Base.cuh"
#include <DataStructures/NodeGroups.h>

#define SPAWN_COUNT uint(64)
#define ELEMENT_SLOT_COUNT uint(96)


namespace ss
{
    __constant__ uint offsets_length;
    __device__ vec4* offsets;

    __shared__ extern PatchElem patchElems[];
    __shared__ float qualities[SPAWN_COUNT];
    __constant__ float MoveCoeff = 0.10;
}

using namespace ss;


// Smoothing Helper
__device__ float computeLocalElementSize(uint vId);
__device__ void accumulatePatchQuality(
        double& patchQuality,
        double& patchWeight,
        double elemQuality);
__device__ float finalizePatchQuality(
        double patchQuality,
        double patchWeight);


// ENTRY POINT //
__device__ void spawnSearchSmoothVert(uint vId)
{
    uint lId = threadIdx.x;

    Topo topo = topos[vId];
    uint neigElemCount = topo.neigElemCount;    
    uint firstLoad = (neigElemCount * lId) / blockDim.x;
    uint lastLoad = (neigElemCount * (lId+1)) / blockDim.x;

    for(uint e = firstLoad; e < lastLoad; ++e)
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

    // Compute local element size
    float localSize = computeLocalElementSize(vId);
    float scale = localSize * MoveCoeff;

    vec4 offset = offsets[lId];

    for(int iter=0; iter < 2; ++iter)
    {
        __syncthreads();

        vec3 spawnPos = verts[vId].p + vec3(offset) * scale;


        double patchWeight = 0.0;
        double patchQuality = 0.0;
        for(uint i=0; i < neigElemCount; ++i)
        {
            vec3 vertPos[] = {
                patchElems[i].p[0],
                patchElems[i].p[1],
                patchElems[i].p[2],
                patchElems[i].p[3],
                patchElems[i].p[4],
                patchElems[i].p[5],
                patchElems[i].p[6],
                patchElems[i].p[7]
            };

            vertPos[patchElems[i].n] = spawnPos;

            float qual = 0.0;
            switch(patchElems[i].type)
            {
            case TET_ELEMENT_TYPE :
                qual = (*tetQualityImpl)(vertPos, patchElems[i].tet);
                break;
            case PRI_ELEMENT_TYPE :
                qual = (*priQualityImpl)(vertPos, patchElems[i].pri);
                break;
            case HEX_ELEMENT_TYPE :
                qual = (*hexQualityImpl)(vertPos, patchElems[i].hex);
                break;
            }

            accumulatePatchQuality(
                patchQuality, patchWeight,
                double(qual));
        }

        qualities[lId] = finalizePatchQuality(patchQuality, patchWeight);

        __syncthreads();

        if(lId == 0)
        {
            uint bestLoc = 0;
            float bestQual = qualities[0];

            for(int i=1; i < SPAWN_COUNT; ++i)
            {
                if(qualities[i] > bestQual)
                {
                    bestLoc = i;
                    bestQual = qualities[i];
                }
            }

            // Update vertex's position
            verts[vId].p += vec3(offsets[bestLoc]) * scale;
        }

        scale /= 3.0;
    }
}

__global__ void smoothSpawnVerticesCudaMain()
{
    if(blockIdx.x < GroupSize)
    {
        uint idx = GroupBase + blockIdx.x;
        uint vId = groupMembers[idx];
        spawnSearchSmoothVert(vId);
    }
}


// CUDA Drivers
void setupCudaIndependentDispatch(const NodeGroups::GpuDispatch& dispatch);

size_t d_offsetsLength = 0;
vec4* d_offsets = nullptr;
void installCudaSpawnSearchSmoother(float moveCoeff,
        const std::vector<glm::vec4> offsetsBuff)
{
    cudaMemcpyToSymbol(MoveCoeff, &moveCoeff, sizeof(float));

    // Main function is directly calling spawnSearchSmoothVert

//    smoothVertFct d_smoothVert = nullptr;
//    cudaMemcpyFromSymbol(&d_smoothVert, spawnSearchSmoothVertPtr, sizeof(smoothVertFct));
//    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));


    uint offsetsLength = offsetsBuff.size();
    size_t offsetsBuffSize = sizeof(decltype(offsetsBuff.front())) * offsetsLength;
    if(d_offsets == nullptr || d_offsetsLength != offsetsLength)
    {
        cudaFree(d_offsets);
        if(!offsetsLength) d_offsets = nullptr;
        else cudaMalloc(&d_offsets, offsetsBuffSize);
        cudaMemcpyToSymbol(offsets, &d_offsets, sizeof(d_offsets));

        d_offsetsLength = offsetsLength;
        cudaMemcpyToSymbol(offsets_length, &offsetsLength, sizeof(uint));
    }

    cudaMemcpy(d_offsets, offsetsBuff.data(), offsetsBuffSize, cudaMemcpyHostToDevice);

    cudaCheckErrors("Spawn offsets update");


    if(verboseCuda)
        printf("I -> CUDA \tSpawn Search smoother installed\n");
}

void smoothCudaSpawnVertices(
        const NodeGroups::GpuDispatch& dispatch)
{
    setupCudaIndependentDispatch(dispatch);

    size_t sharedDim = sizeof(PatchElem) * ELEMENT_SLOT_COUNT;

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothSpawnVerticesCudaMain<<<dispatch.workgroupCount, SPAWN_COUNT, sharedDim>>>();
    cudaCheckErrors("CUDA error during vertices smoothing");

    cudaDeviceSynchronize();
}
