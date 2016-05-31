#include "Base.cuh"

__device__ vec3 shellSnapToBoundary(int boundaryID, vec3 pos)
{
    vec3 npos = normalize(pos);
    if(boundaryID == 2)
        return npos * 0.5f;
    else
        return npos;
}


__device__ snapToBoundaryFct shellSnapToBoundaryPtr = shellSnapToBoundary;


// CUDA Drivers
void installCudaShellBoundary()
{
    snapToBoundaryFct d_snapToBoundary = nullptr;
    cudaMemcpyFromSymbol(&d_snapToBoundary, shellSnapToBoundaryPtr, sizeof(snapToBoundaryFct));
    cudaMemcpyToSymbol(snapToBoundary, &d_snapToBoundary, sizeof(snapToBoundaryFct));


    if(verboseCuda)
        printf("I -> CUDA \tShell boundary installed\n");
}
