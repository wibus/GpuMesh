#include "Base.cuh"

__device__ vec3 sphereSnapToBoundary(int boundaryID, vec3 pos)
{
    vec3 npos = normalize(pos);
    if(boundaryID == 1)
        return npos * 0.5f;
    else
        return npos;
}


__device__ snapToBoundaryFct sphereSnapToBoundaryPtr = sphereSnapToBoundary;


// CUDA Drivers
void installCudaSphereBoundary()
{
    snapToBoundaryFct d_snapToBoundary = nullptr;
    cudaMemcpyFromSymbol(&d_snapToBoundary, sphereSnapToBoundaryPtr, sizeof(snapToBoundaryFct));
    cudaMemcpyToSymbol(snapToBoundary, &d_snapToBoundary, sizeof(snapToBoundaryFct));


    if(verboseCuda)
        printf("I -> CUDA \tSphere boundary installed\n");
}
