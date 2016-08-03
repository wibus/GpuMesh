#include "Base.cuh"
#include <DataStructures/NodeGroups.h>

#define SPAWN_COUNT uint(64)


struct TetVert
{
    __device__ TetVert() {}
    float3 p[TET_VERTEX_COUNT];
};


__constant__ uint offsets_length;
__device__ vec4* offsets;

__shared__ Tet tetElems[SPAWN_COUNT];
__shared__ TetVert tetVerts[SPAWN_COUNT];
__shared__ float qualities[SPAWN_COUNT];
__constant__ float SSMoveCoeff = 0.10;


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
__device__ inline float3 toFloat3(const vec3& v)
{
    return make_float3(v.x, v.y, v.z);
}

__device__ inline vec3 toVec3(const float3& v)
{
    return vec3(v.x, v.y, v.z);
}

__device__ void spawnSearchSmoothVert(uint vId)
{
    uint lId = threadIdx.x;
    Topo topo = topos[vId];

    uint neigElemCount = topo.neigElemCount;

    if(lId < neigElemCount)
    {
        NeigElem elem = neigElems[topo.neigElemBase + lId];

        Tet tet = tets[elem.id];
        tetElems[lId] = tet;

        tetVerts[lId].p[0] = toFloat3(verts[tet.v[0]].p);
        tetVerts[lId].p[1] = toFloat3(verts[tet.v[1]].p);
        tetVerts[lId].p[2] = toFloat3(verts[tet.v[2]].p);
        tetVerts[lId].p[3] = toFloat3(verts[tet.v[3]].p);
    }

    __syncthreads();

    // Compute local element size
    float localSize = computeLocalElementSize(vId);
    float scale = localSize * SSMoveCoeff;

    vec4 offset = offsets[lId];
    vec3 spawnPos = verts[vId].p + vec3(offset) * scale;


    double patchWeight = 0.0;
    double patchQuality = 1.0;
    for(uint i=0; i < neigElemCount; ++i)
    {
        vec3 tetVert[4] = {
            toVec3(tetVerts[i].p[0]),
            toVec3(tetVerts[i].p[1]),
            toVec3(tetVerts[i].p[2]),
            toVec3(tetVerts[i].p[3]),
        };

        Tet tetElem = tetElems[i];

        if(tetElem.v[0] == vId)
            tetVert[0] = spawnPos;
        else if(tetElem.v[1] == vId)
            tetVert[1] = spawnPos;
        else if(tetElem.v[2] == vId)
            tetVert[2] = spawnPos;
        else if(tetElem.v[3] == vId)
            tetVert[3] = spawnPos;

        accumulatePatchQuality(
            patchQuality, patchWeight,
            double((*tetQualityImpl)(tetVert, tetElem)));
    }

    qualities[lId] = finalizePatchQuality(patchQuality, patchWeight);

    __syncthreads();

    if(lId == 0)
    {
        uint bestLoc = 0;
        float bestQual = -1.0/0.0; // -Inf

        for(int i=0; i < SPAWN_COUNT; ++i)
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
    cudaMemcpyToSymbol(SSMoveCoeff, &moveCoeff, sizeof(float));

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

    cudaCheckErrors("CUDA error before vertices smoothing");
    smoothSpawnVerticesCudaMain<<<dispatch.gpuBufferSize, SPAWN_COUNT>>>();
    cudaCheckErrors("CUDA error during vertices smoothing");

    cudaDeviceSynchronize();
}
