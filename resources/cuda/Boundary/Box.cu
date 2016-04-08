#include "Base.cuh"

__device__ vec3 boxSnapToBoundary(int boundaryID, vec3 pos)
{
    pos[boundaryID-1] /= abs(pos[boundaryID-1]);
    return pos;
}

__device__ snapToBoundaryFct boxSnapToBoundaryPtr = boxSnapToBoundary;


// CUDA Drivers
void installCudaBoxBoundary()
{
    snapToBoundaryFct d_snapToBoundary = nullptr;
    cudaMemcpyFromSymbol(&d_snapToBoundary, boxSnapToBoundaryPtr, sizeof(snapToBoundaryFct));
    cudaMemcpyToSymbol(snapToBoundary, &d_snapToBoundary, sizeof(snapToBoundaryFct));


    if(verboseCuda)
        printf("I -> CUDA \tBox boundary installed\n");
}
