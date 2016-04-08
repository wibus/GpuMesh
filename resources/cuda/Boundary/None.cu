#include "Base.cuh"

__device__ vec3 noneSnapToBoundary(int boundaryID, vec3 pos)
{
    return pos;
}


__device__ snapToBoundaryFct noneSnapToBoundaryPtr = noneSnapToBoundary;


// CUDA Drivers
void installCudaNoneBoundary()
{
    snapToBoundaryFct d_snapToBoundary = nullptr;
    cudaMemcpyFromSymbol(&d_snapToBoundary, noneSnapToBoundaryPtr, sizeof(snapToBoundaryFct));
    cudaMemcpyToSymbol(snapToBoundary, &d_snapToBoundary, sizeof(snapToBoundaryFct));


    if(verboseCuda)
        printf("I -> CUDA \tNone boundary installed\n");
}
