#include "Base.cuh"

#include <vector>


struct VertexAccum
{
    vec3 posAccum;
    float weightAccum;
};

__device__ uint vertexAccums_length;
__device__ VertexAccum* vertexAccums = nullptr;


__device__ void addPosition(uint vId, vec3 pos, float weight)
{
    vec3 val = pos * weight;
    atomicAdd(&vertexAccums[vId].posAccum.x, val.x);
    atomicAdd(&vertexAccums[vId].posAccum.y, val.y);
    atomicAdd(&vertexAccums[vId].posAccum.z, val.z);
    atomicAdd(&vertexAccums[vId].weightAccum, weight);
}

__device__ bool assignAverage(uint vId, vec3& pos)
{
    float weightAccum = vertexAccums[vId].weightAccum;
    if(weightAccum > 0.0)
    {
        pos = vertexAccums[vId].posAccum / weightAccum;
        return true;
    }
    return false;
}

__device__ void reinitAccum(uint vId)
{
    vertexAccums[vId].posAccum = vec3(0.0);
    vertexAccums[vId].weightAccum  = 0.0;
}


// CUDA Drivers
size_t d_vertexAccumsLength = 0;
uint* d_vertexAccums = nullptr;
void vertexAccumCudaInstall(size_t vertCount)
{
    std::vector<VertexAccum> vertexAccumsBuff(vertCount);

    // Group members
    uint vertexAccumsLength = vertexAccumsBuff.size();
    size_t vertexAccumsBuffSize = sizeof(decltype(vertexAccumsBuff.front())) * vertexAccumsLength;
    if(d_vertexAccums == nullptr || d_vertexAccumsLength != vertexAccumsLength)
    {
        cudaFree(d_vertexAccums);
        if(!vertexAccumsLength) d_vertexAccums = nullptr;
        else cudaMalloc(&d_vertexAccums, vertexAccumsBuffSize);
        cudaMemcpyToSymbol(vertexAccums, &d_vertexAccums, sizeof(d_vertexAccums));

        d_vertexAccumsLength = vertexAccumsLength;
        cudaMemcpyToSymbol(vertexAccums_length, &vertexAccumsLength, sizeof(uint));
    }

    cudaMemcpy(d_vertexAccums, vertexAccumsBuff.data(), vertexAccumsBuffSize, cudaMemcpyHostToDevice);
    printf("I -> CUDA \tVertex Accums updated\n");
}

void vertexAccumCudaDeinstall()
{
    vertexAccumCudaInstall(0);
}
