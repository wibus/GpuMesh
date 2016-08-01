#include "Base.cuh"

#define PROPOSITION_COUNT uint(64)

__device__  float SSMoveCoeff = 0.10;


// Smoothing helper
__device__ float patchQuality(uint vId);


// ENTRY POINT //
__device__ void spawnSearchSmoothVert(uint vId)
{
}

__device__ smoothVertFct spawnSearchSmoothVertPtr = spawnSearchSmoothVert;


// CUDA Drivers
void installCudaSpawnSearchSmoother(float moveCoeff)
{
    cudaMemcpyToSymbol(SSMoveCoeff, &moveCoeff, sizeof(float));

    smoothVertFct d_smoothVert = nullptr;
    cudaMemcpyFromSymbol(&d_smoothVert, spawnSearchSmoothVertPtr, sizeof(smoothVertFct));
    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));


    if(verboseCuda)
        printf("I -> CUDA \tSpawn Search smoother installed\n");
}
