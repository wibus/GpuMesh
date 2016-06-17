#include "Base.cuh"


// ENTRY POINT //
__device__ void springLaplaceSmoothVert(uint vId)
{
    vec3 patchCenter = computeVertexEquilibrium(vId);
    verts[vId].p = mix(verts[vId].p, patchCenter, MoveCoeff);
}

__device__ smoothVertFct springLaplaceSmoothVertPtr = springLaplaceSmoothVert;


// CUDA Drivers
void installCudaSpringLaplaceSmoother()
{
    smoothVertFct d_smoothVert = nullptr;
    cudaMemcpyFromSymbol(&d_smoothVert, springLaplaceSmoothVertPtr, sizeof(smoothVertFct));
    cudaMemcpyToSymbol(smoothVert, &d_smoothVert, sizeof(smoothVertFct));


    if(verboseCuda)
        printf("I -> CUDA \tSpring Laplace smoother installed\n");
}
