#include "Base.cuh"

__device__ vec3 freeSnapToBoundary(int boundaryID, vec3 pos)
{
    return pos;
}


__device__ snapToBoundaryFct freeSnapToBoundaryPtr = freeSnapToBoundary;


// CUDA Drivers
void installCudaBoundaryFree()
{
    snapToBoundaryFct d_snapToBoundary = nullptr;
    cudaMemcpyFromSymbol(&d_snapToBoundary, freeSnapToBoundaryPtr, sizeof(snapToBoundaryFct));
    cudaMemcpyToSymbol(snapToBoundary, &d_snapToBoundary, sizeof(snapToBoundaryFct));


    if(verboseCuda)
        printf("I -> CUDA \tBoundary free installed\n");
}
